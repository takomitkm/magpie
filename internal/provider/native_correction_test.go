package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/settings"
)

const nativeMessagesList = `{"object":"list","data":[{"id":"zcode/GLM-5.3-Flash","native_endpoints":["/v1/messages"],"reasoning":true,"context_window":1000000}]}`
const nativeMessagesAnswer = `{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn"}`

// The reported relay accepts Chat, but advertises its model on Messages.
// A direct provider/model request used to work while its only routing
// group's member was struck out by NoAPI. Fetch adds a verified sibling
// and remembers the model's API while preserving the configured base.
func TestNativeCorrectionRestoresRoutingGroup(t *testing.T) {
	for _, prefix := range []string{"", "/gateway"} {
		t.Run(prefix, func(t *testing.T) {
			detectHome(t)
			var probes atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.Method + " " + r.URL.Path {
				case "GET " + prefix + "/v1/models":
					io.WriteString(w, nativeMessagesList)
				case "POST " + prefix + "/v1/chat/completions":
					io.WriteString(w, `{"choices":[{"message":{"content":"OK"}}]}`)
				case "POST " + prefix + "/v1/messages":
					probes.Add(1)
					if r.Header.Get("x-api-key") != "fake-key" || r.Header.Get("anthropic-version") != "2023-06-01" {
						t.Error("native probe lost the provider's signing")
					}
					var body map[string]any
					json.NewDecoder(r.Body).Decode(&body)
					if body["model"] != "zcode/GLM-5.3-Flash" {
						t.Errorf("native model = %v", body["model"])
					}
					io.WriteString(w, nativeMessagesAnswer)
				default:
					http.NotFound(w, r)
				}
			}))
			defer srv.Close()
			root := srv.URL + prefix
			p := Provider{ID: "relay", Name: "Relay", Key: "fake-key", Chat: root + "/v1", BaseAPI: "chat", Models: []string{"zcode/GLM-5.3-Flash"}, Unlisted: true, Proxy: "direct"}
			if err := Save(p); err != nil {
				t.Fatal(err)
			}
			if err := SaveGroup(Group{ID: "native", Name: "Native", Members: []string{"relay/zcode/GLM-5.3-Flash"}}); err != nil {
				t.Fatal(err)
			}
			p = mustNativeProvider(t, p.ID)
			if _, err := p.Fetch(context.Background()); err != nil {
				t.Fatal(err)
			}
			q := mustNativeProvider(t, p.ID)
			if q.Chat != p.Chat || q.Responses != "" || q.Anthropic != root || q.BaseAPI != p.BaseAPI {
				t.Fatalf("native API was not corrected: chat=%q responses=%q anthropic=%q baseAPI=%q", q.Chat, q.Responses, q.Anthropic, q.BaseAPI)
			}
			if !q.Unlisted || q.Key != p.Key || q.Proxy != p.Proxy || !slices.Equal(q.Models, p.Models) {
				t.Fatal("native correction changed provider choices or credentials")
			}
			if q.NoAPIFor(p.Models[0]) || len(q.Exposed()) != 1 {
				t.Fatal("native model is still excluded from the provider")
			}
			_, members, ok := FindGroup("group/native")
			if !ok || len(members) != 1 || members[0].Provider.Anthropic != root {
				t.Fatalf("routing group still has no native member: ok=%v members=%v", ok, members)
			}
			if _, err := q.Fetch(context.Background()); err != nil {
				t.Fatal(err)
			}
			if probes.Load() != 1 {
				t.Fatalf("corrected provider was probed %d times", probes.Load())
			}
		})
	}
}

func TestNativeCorrectionFirstTestUsesCorrectedAPI(t *testing.T) {
	detectHome(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method + " " + r.URL.Path {
		case "GET /v1/models":
			io.WriteString(w, nativeMessagesList)
		case "POST /v1/messages":
			io.WriteString(w, nativeMessagesAnswer)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake-key", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"zcode/GLM-5.3-Flash"}}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	results := p.Test(context.Background())
	if !slices.ContainsFunc(results, func(r Result) bool { return r.OK && r.Protocol == Anthropic }) {
		t.Fatalf("first Test used the stale API: %+v", results)
	}
}

func mustNativeProvider(t *testing.T, id string) Provider {
	t.Helper()
	p, err := Find(id)
	if err != nil {
		t.Fatal(err)
	}
	return *p
}

func TestNativeCorrectionKeepsUnverifiedConfiguration(t *testing.T) {
	for _, reply := range []struct {
		name   string
		status int
		body   string
	}{
		{"refused", 403, `{"error":{"message":"forbidden"}}`},
		{"missing", 404, `not found`},
		{"login page", 200, `<html>sign in</html>`},
		{"wrapped error", 200, `{"error":{"message":"refused"}}`},
		{"false success", 200, `{"success":false,"type":"message","content":[]}`},
		{"error with answer", 200, `{"error":{"message":"refused"},"type":"message","content":[]}`},
		{"truncated answer", 200, `{"type":"message","content":[`},
		{"wrong protocol", 200, `{"choices":[{"message":{"content":"OK"}}]}`},
	} {
		t.Run(reply.name, func(t *testing.T) {
			detectHome(t)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodGet {
					io.WriteString(w, nativeMessagesList)
					return
				}
				w.WriteHeader(reply.status)
				io.WriteString(w, reply.body)
			}))
			defer srv.Close()
			p := Provider{ID: "relay", Name: "Relay", Key: "fake-key", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"zcode/GLM-5.3-Flash"}}
			if err := Save(p); err != nil {
				t.Fatal(err)
			}
			p = mustNativeProvider(t, p.ID)
			if _, err := p.Fetch(context.Background()); err != nil {
				t.Fatal(err)
			}
			q := mustNativeProvider(t, p.ID)
			if q.Chat != p.Chat || q.Anthropic != "" || q.BaseAPI != p.BaseAPI {
				t.Fatal("an unverified endpoint replaced the configured API")
			}
		})
	}
}

func TestNativeCorrectionNeverFollowsRedirect(t *testing.T) {
	detectHome(t)
	var reached atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached.Add(1)
		io.WriteString(w, nativeMessagesAnswer)
	}))
	defer target.Close()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			io.WriteString(w, nativeMessagesList)
			return
		}
		http.Redirect(w, r, target.URL+"/v1/messages", http.StatusTemporaryRedirect)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake-key", Chat: srv.URL + "/v1", Models: []string{"zcode/GLM-5.3-Flash"}}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	if reached.Load() != 0 || mustNativeProvider(t, p.ID).Anthropic != "" {
		t.Fatal("native correction followed a redirect or accepted it")
	}
}

func TestNativeCorrectionRejectsSameOriginRedirect(t *testing.T) {
	detectHome(t)
	var redirected atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/messages" {
			http.Redirect(w, r, "/another-api", http.StatusTemporaryRedirect)
			return
		}
		redirected.Add(1)
		io.WriteString(w, nativeMessagesAnswer)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake-key", Anthropic: srv.URL, BaseAPI: "anthropic"}
	if p.probeNative(context.Background(), "zcode/GLM-5.3-Flash") || redirected.Load() != 0 {
		t.Fatal("native probe followed a same-origin redirect")
	}
}

func TestNativeCorrectionKeepsUsageOnly(t *testing.T) {
	detectHome(t)
	var posts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			posts.Add(1)
			io.WriteString(w, nativeMessagesAnswer)
			return
		}
		io.WriteString(w, nativeMessagesList)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake-key", Chat: srv.URL + "/v1", BaseAPI: "chat", NoModels: true}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	q := mustNativeProvider(t, p.ID)
	if posts.Load() != 0 || q.Chat != p.Chat || q.Anthropic != "" || !q.ServesNone() {
		t.Fatal("usage-only refresh generated a reply or changed its protocol")
	}
}

func TestNativeCorrectionDoesNotOverwriteConcurrentEdit(t *testing.T) {
	detectHome(t)
	var saved Provider
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			io.WriteString(w, nativeMessagesList)
			return
		}
		q := saved
		q.Name = "Edited while detecting"
		if err := Save(q); err != nil {
			t.Error(err)
		}
		io.WriteString(w, nativeMessagesAnswer)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake-key", Chat: srv.URL + "/v1", Models: []string{"zcode/GLM-5.3-Flash"}}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	saved = mustNativeProvider(t, p.ID)
	if _, err := saved.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	q := mustNativeProvider(t, p.ID)
	if q.Name != "Edited while detecting" || q.Anthropic != "" || q.Chat != saved.Chat {
		t.Fatal("native correction overwrote a concurrent edit")
	}
}

func TestNativeCorrectionNeverProbesDisabledModels(t *testing.T) {
	for _, all := range []bool{true, false} {
		t.Run(fmt.Sprint(all), func(t *testing.T) {
			detectHome(t)
			var posts atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodGet {
					io.WriteString(w, `{"data":[{"id":"a","native_endpoints":["/v1/messages"]},{"id":"b","native_endpoints":["/v1/messages"]}]}`)
					return
				}
				posts.Add(1)
				var body map[string]any
				json.NewDecoder(r.Body).Decode(&body)
				if body["model"] != "b" {
					t.Errorf("probed disabled model: %v", body["model"])
				}
				io.WriteString(w, nativeMessagesAnswer)
			}))
			defer srv.Close()
			p := Provider{ID: "relay", Name: "Relay", Key: "fake-key", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"a", "b"}}
			if err := Save(p); err != nil {
				t.Fatal(err)
			}
			patterns := []string{"relay/a"}
			if all {
				patterns = append(patterns, "relay/b")
			}
			if err := SetDisabledModels(patterns); err != nil {
				t.Fatal(err)
			}
			p = mustNativeProvider(t, p.ID)
			if _, err := p.Fetch(context.Background()); err != nil {
				t.Fatal(err)
			}
			q := mustNativeProvider(t, p.ID)
			if all && (posts.Load() != 0 || q.Chat != p.Chat) {
				t.Fatal("all-disabled provider was probed or corrected")
			}
			if !all && (posts.Load() != 1 || q.Native("b") != Anthropic || q.BaseAPI != p.BaseAPI) {
				t.Fatal("enabled model was not used to verify the native API")
			}
		})
	}
}

func TestNativeCorrectionUsesLocalModelID(t *testing.T) {
	detectHome(t)
	s := settings.Load()
	s.ModelIDs = map[string]string{"relay/fast": "zcode/GLM-5.3-Flash"}
	if err := settings.Save(s); err != nil {
		t.Fatal(err)
	}
	var posts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			io.WriteString(w, nativeMessagesList)
			return
		}
		posts.Add(1)
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["model"] != "zcode/GLM-5.3-Flash" {
			t.Errorf("wire model = %v", body["model"])
			http.Error(w, "unknown model", 404)
			return
		}
		io.WriteString(w, nativeMessagesAnswer)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake-key", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"fast"}}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	q := mustNativeProvider(t, p.ID)
	if posts.Load() != 1 || q.Native("fast") != Anthropic || q.BaseAPI != p.BaseAPI || !slices.Equal(q.Models, []string{"fast"}) {
		t.Fatal("native correction lost the local model ID or sent it upstream")
	}
}

func TestNativeCorrectionInfersOnlyStandardSiblings(t *testing.T) {
	p := Provider{Chat: "https://relay.example/gateway/v1", BaseAPI: "chat"}
	for _, tc := range []struct {
		name   string
		change func(*Provider)
	}{
		{"independent endpoint", func(p *Provider) { p.Responses = "https://other.example/v1" }},
		{"protocol key", func(p *Provider) { p.KeyProtocol = Chat }},
		{"custom path", func(p *Provider) { p.Chat = "https://relay.example/custom" }},
		{"query", func(p *Provider) { p.Chat += "?tenant=x" }},
		{"userinfo", func(p *Provider) { p.Chat = "https://user:pass@relay.example/v1" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			q := p
			tc.change(&q)
			if _, ok := q.inferredEndpoint(Anthropic); ok {
				t.Fatal("inferred an ambiguous sibling")
			}
		})
	}
	q, ok := p.inferredEndpoint(Anthropic)
	if !ok || q.Chat != p.Chat || q.BaseAPI != p.BaseAPI || q.Anthropic != "https://relay.example/gateway" {
		t.Fatalf("sibling changed original configuration: %+v", q)
	}
}
