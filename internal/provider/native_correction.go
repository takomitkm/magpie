package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/probelog"
)

// nativeCorrection replaces a custom provider's single API when every
// selected model is listed on one other API. A gateway's Chat adapter may
// answer even though its list names Messages alone: keep the configuration
// on that native API, so Exposed and routing groups agree with direct calls.
// Independent endpoints, protocol-bound keys and model overrides are not
// inferred from another endpoint's list.
func (p Provider) nativeCorrection(ms []catalog.Model) (Provider, string, bool) {
	if p.Account != nil || p.Preset != "" || p.Decide != "" || p.ModelsURL != "" || len(p.Keys) > 0 || p.KeyProtocol != "" || p.NativeCompaction || p.ServesNone() || p.Off {
		return p, "", false
	}
	speaks := p.Speaks()
	if len(speaks) != 1 || !slices.Contains(Protocols, speaks[0]) || p.BaseAPI != "" && p.BaseAPI != string(speaks[0]) {
		return p, "", false
	}
	old := speaks[0]
	base := p.Base(old)
	u, err := url.Parse(base)
	if err != nil || u.Host == "" || u.Scheme != "http" && u.Scheme != "https" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return p, "", false
	}
	root := strings.TrimRight(base, "/")
	if old != Anthropic {
		var ok bool
		root, ok = strings.CutSuffix(root, "/v1")
		if !ok {
			return p, "", false // no convention for a vendor's custom API path
		}
	}
	listed := make(map[string]catalog.Model, len(ms))
	for _, m := range catalog.Chat(ms) {
		listed[m.ID] = m
	}
	picks := p.Models
	if len(picks) == 0 {
		for _, m := range catalog.Chat(ms) {
			picks = append(picks, m.ID)
		}
	}
	var target Protocol
	model := ""
	off := disabledNow()
	for _, id := range picks {
		if _, set := p.apiSet(id); set {
			return p, "", false
		}
		m, ok := listed[p.OwnModelID(id)]
		if !ok || len(m.APIs) != 1 {
			return p, "", false
		}
		pr := Protocol(m.APIs[0])
		if pr == old || !slices.Contains(Protocols, pr) || target != "" && target != pr {
			return p, "", false
		}
		target = pr
		if model == "" && !off(p.ID, id) {
			model = id
		}
	}
	if target == "" || model == "" {
		return p, "", false
	}
	q := p
	q.Chat, q.Responses, q.Anthropic = "", "", ""
	switch target {
	case Chat:
		q.Chat = root + "/v1"
	case Responses:
		q.Responses = root + "/v1"
	case Anthropic:
		q.Anthropic = root
	}
	q.BaseAPI = string(target)
	return q, model, true
}

// correctNative runs only after a successful model-list fetch. The one
// inferred endpoint must answer the selected model before it replaces the
// old API. Neither a failed probe nor an unsaved editor's fetch writes
// providers.json; an edit made while the probe was out wins over it.
func (p Provider) correctNative(ctx context.Context, ms []catalog.Model) error {
	q, model, ok := p.nativeCorrection(ms)
	if !ok {
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
	if !q.probeNative(ctx, model) || ctx.Err() != nil {
		return nil
	}
	f, err = read()
	if err != nil {
		return err
	}
	i = slices.IndexFunc(f.Providers, func(x Provider) bool { return x.ID == p.ID })
	if i < 0 || !sameStored(f.Providers[i], p) {
		return nil
	}
	return Save(q)
}

// probeNative asks only the inferred API, on the configured origin and
// deployment prefix. A redirect is not evidence that this endpoint works,
// and a login page or an error wrapped in HTTP 200 is not a model answer.
func (p Provider) probeNative(ctx context.Context, model string) bool {
	if _, off := ModelDisabled(p.ID, model); off {
		return false
	}
	ctx, cancel := context.WithTimeout(p.ownRequests(p.Via(ctx), probelog.KindDetect), testWait)
	defer cancel()
	pr := Protocol(p.BaseAPI)
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
		Type    string            `json:"type"`
		Object  string            `json:"object"`
		Content []json.RawMessage `json:"content"`
		Choices []json.RawMessage `json:"choices"`
		Output  []json.RawMessage `json:"output"`
	}
	if json.NewDecoder(io.LimitReader(res.Body, 64<<10)).Decode(&answer) != nil || len(answer.Error) > 0 && string(answer.Error) != "null" {
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
