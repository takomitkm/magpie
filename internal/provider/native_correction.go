package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/probelog"
)

type backgroundListKey struct{}

// WithBackgroundList marks a list fetch that magpie makes on its own
// schedule (KeepListsFresh, a Remote magpie's list followed): the models
// are refreshed, but nothing that costs a generation is verified.
func WithBackgroundList(ctx context.Context) context.Context {
	return context.WithValue(ctx, backgroundListKey{}, true)
}

func backgroundList(ctx context.Context) bool {
	on, _ := ctx.Value(backgroundListKey{}).(bool)
	return on
}

// inferredEndpoint derives only the standard sibling on the same origin and
// deployment prefix. Independent endpoints and nonstandard paths stay as set.
func (p Provider) inferredEndpoint(target Protocol) (Provider, bool) {
	if !slices.Contains(Protocols, target) || p.Base(target) != "" || p.KeyProtocol != "" || p.NativeCompaction || p.ModelsURL != "" {
		return p, false
	}
	root := ""
	for _, pr := range p.Speaks() {
		if !slices.Contains(Protocols, pr) {
			return p, false
		}
		base := p.Base(pr)
		u, err := url.Parse(base)
		if err != nil || u.Host == "" || u.Scheme != "http" && u.Scheme != "https" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
			return p, false
		}
		r := strings.TrimRight(base, "/")
		if pr != Anthropic {
			var ok bool
			r, ok = strings.CutSuffix(r, "/v1")
			if !ok {
				return p, false
			}
		}
		if root != "" && root != r {
			return p, false
		}
		root = r
	}
	if root == "" {
		return p, false
	}
	q := p
	switch target {
	case Chat:
		q.Chat = root + "/v1"
	case Responses:
		q.Responses = root + "/v1"
	case Anthropic:
		q.Anthropic = root
	}
	q.NativeInferred = maps.Clone(p.NativeInferred)
	if q.NativeInferred == nil {
		q.NativeInferred = map[Protocol]string{}
	}
	q.NativeInferred[target] = q.Base(target)
	return q, true
}

// correctNative validates each enabled model separately, preferring its
// declaration and falling back to the originally configured APIs. Successful
// sibling endpoints are added, never substituted for the user's URL/BaseAPI.
// The shared generation budget and total deadline bound a large model list.
//
// Only a model whose own list entry declares an API is verified: a vendor
// that says nothing about a model has not asked for the repair, and paying a
// generation for every model of every list refresh is not magpie's to spend
// (OpenRouter lists hundreds with no per-model declaration, and a fetch asks
// several URLs already). A background refresh (KeepListsFresh) verifies
// nothing either: its answer is already known from the last one, and a list
// tick is not the user asking.
func (p Provider) correctNative(ctx context.Context, ms []catalog.Model) error {
	if p.Account != nil || p.Preset != "" || p.Decide != "" || len(p.Keys) > 0 || p.KeyProtocol != "" || p.NativeCompaction || p.ServesNone() || p.Off {
		return nil
	}
	// A provider given a list URL of its own is listed where its user said,
	// and its bases are not asked anything beyond it.
	if p.ModelsURL != "" || backgroundList(ctx) {
		return nil
	}
	f, err := read()
	if err != nil {
		return err
	}
	i := slices.IndexFunc(f.Providers, func(x Provider) bool { return x.ID == p.ID })
	if i < 0 || !sameStored(f.Providers[i], p) {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	listed := map[string]catalog.Model{}
	for _, m := range catalog.Chat(ms) {
		listed[m.ID] = m
	}
	picks := slices.Clone(p.Models)
	if len(picks) == 0 {
		for _, m := range p.ids().list(catalog.Chat(ms)) {
			picks = append(picks, m.ID)
		}
	}
	keys := p.KeysOn()
	if len(keys) == 0 && p.Key == "" {
		keys = []KeyAccount{p.first()}
	}
	type success struct {
		model string
		key   KeyAccount
		proto Protocol
	}
	var successes []success
	var failures []success
	identities := map[string][32]byte{}
	q := p
	budget := 24
	for _, model := range picks {
		if ctx.Err() != nil || budget == 0 {
			break
		}
		if _, off := ModelDisabled(p.ID, model); off {
			continue
		}
		if _, set := p.apiSet(model); set {
			continue
		}
		m, listedHere := listed[p.OwnModelID(model)]
		if !listedHere {
			continue
		}
		// No declaration, no repair: the list said nothing about this model,
		// so there is nothing to reconcile and nothing to pay for.
		if len(m.APIs) == 0 {
			continue
		}
		manualMatch, inferredDeclaration := false, false
		for _, api := range m.APIs {
			proto := Protocol(api)
			if p.inferredAPI(proto) {
				inferredDeclaration = true
			} else if p.Base(proto) != "" {
				manualMatch = true
			}
		}
		// A declaration already served at a hand-configured endpoint needs
		// no repair and no paid generation during an ordinary list refresh.
		if manualMatch && !inferredDeclaration {
			continue
		}
		identity, ok := p.learningKey(model)
		if !ok {
			continue
		}
		identities[model] = identity
		for _, key := range keys {
			if ctx.Err() != nil || budget == 0 {
				break
			}
			if len(m.Keys) > 0 && !slices.Contains(m.Keys, keyID(key.Key)) {
				continue
			}
			view := q.WithKey(key)
			if v, ok := view.learned(model); ok && time.Now().Before(v.Until) {
				continue
			}
			var candidates []Protocol
			for _, a := range m.APIs {
				pr := Protocol(a)
				if slices.Contains(Protocols, pr) && !slices.Contains(candidates, pr) {
					candidates = append(candidates, pr)
				}
			}
			// No declaration means no sibling guesses: only configured APIs may
			// be tried. A failed declaration may still work through the old adapter.
			for _, pr := range p.WithKey(key).Speaks() {
				if slices.Contains(Protocols, pr) && !slices.Contains(candidates, pr) {
					candidates = append(candidates, pr)
				}
			}
			tried, succeeded := false, false
			for _, pr := range candidates {
				if ctx.Err() != nil || budget == 0 {
					break
				}
				attempt := q.WithKey(key)
				added := false
				if attempt.Base(pr) == "" {
					var ok bool
					attempt, ok = attempt.inferredEndpoint(pr)
					if !ok {
						continue
					}
					added = true
				}
				budget--
				tried = true
				if !attempt.probeNativeAPI(ctx, model, pr) {
					continue
				}
				if added && q.Base(pr) == "" {
					switch pr {
					case Chat:
						q.Chat = attempt.Chat
					case Responses:
						q.Responses = attempt.Responses
					case Anthropic:
						q.Anthropic = attempt.Anthropic
					}
					q.NativeInferred = maps.Clone(q.NativeInferred)
					if q.NativeInferred == nil {
						q.NativeInferred = map[Protocol]string{}
					}
					q.NativeInferred[pr] = q.Base(pr)
				}
				successes = append(successes, success{model, key, pr})
				succeeded = true
				break
			}
			if tried && !succeeded && ctx.Err() == nil {
				failures = append(failures, success{model: model, key: key})
			}
		}
	}
	// Model IDs, wire names and effective signing settings live outside the
	// provider file too. A reply for their old values cannot prove new ones.
	for model, before := range identities {
		after, ok := p.learningKey(model)
		if !ok || before != after {
			return nil
		}
	}
	// An edit while any probe was out wins, including over learned evidence.
	f, err = read()
	if err != nil {
		return err
	}
	i = slices.IndexFunc(f.Providers, func(x Provider) bool { return x.ID == p.ID })
	if i < 0 || !sameStored(f.Providers[i], p) {
		return nil
	}
	if ctx.Err() != nil {
		return nil
	}
	if !sameStored(q, p) {
		if err := Save(q); err != nil {
			return err
		}
		saved, err := Find(q.ID)
		if err != nil {
			return err
		}
		q = *saved
	}
	for _, s := range successes {
		q.WithKey(s.key).RememberAPI(s.model, s.proto)
	}
	for _, s := range failures {
		q.WithKey(s.key).rememberLearning(s.model, "", failedAPITTL)
	}
	return nil
}

// probeNative is kept for callers testing a provider's selected base API.
func (p Provider) probeNative(ctx context.Context, model string) bool {
	return p.probeNativeAPI(ctx, model, Protocol(p.BaseAPI))
}

// probeNativeAPI uses the normal minimal body, model mapping and signing,
// but rejects redirects and requires a complete, protocol-shaped JSON reply.
func (p Provider) probeNativeAPI(ctx context.Context, model string, pr Protocol) bool {
	if _, off := ModelDisabled(p.ID, model); off {
		return false
	}
	ctx, cancel := context.WithTimeout(p.ownRequests(p.Via(ctx), probelog.KindDetect), testWait)
	defer cancel()
	u, body := tiny(p, pr, UpstreamNameAs(p, model, p.OwnModelID(model)))
	ctx, b := p.tested(ctx, u, body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, bytes.NewReader(b))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	if pr == Anthropic {
		req.Header.Set("anthropic-version", "2023-06-01")
	}
	if err := p.Sign(ctx, req, pr, b); err != nil {
		return false
	}
	client := *http.DefaultClient
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	res, err := p.Do(&client, req)
	if err != nil {
		return false
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode >= 300 {
		return false
	}
	var answer struct {
		Error   json.RawMessage   `json:"error"`
		Success *bool             `json:"success"`
		Type    string            `json:"type"`
		Object  string            `json:"object"`
		Content []json.RawMessage `json:"content"`
		Choices []json.RawMessage `json:"choices"`
		Output  []json.RawMessage `json:"output"`
	}
	if json.NewDecoder(io.LimitReader(res.Body, 64<<10)).Decode(&answer) != nil || len(answer.Error) > 0 && string(answer.Error) != "null" || answer.Success != nil && !*answer.Success {
		return false
	}
	switch pr {
	case Anthropic:
		return answer.Type == "message" && answer.Content != nil
	case Chat:
		return len(answer.Choices) > 0
	case Responses:
		return answer.Object == "response" && answer.Output != nil
	}
	return false
}
