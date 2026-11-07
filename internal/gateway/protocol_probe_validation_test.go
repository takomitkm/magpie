package gateway

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/settings"
)

func TestAmbiguousProtocolProbeWireChanged(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		if r.URL.Path == "/v1/responses" {
			st := settings.Load()
			st.ModelWires = map[string]string{"probe/probe-model": "new-wire"}
			if err := settings.Save(st); err != nil {
				t.Error(err)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, probeGood)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(404)
		io.WriteString(w, `{"error":{"message":"`+probeModelRefusal+`"}}`)
	}))
	t.Cleanup(up.Close)
	if err := provider.Save(provider.Provider{ID: "probe", Name: "Probe", Key: "k", Models: []string{"probe-model"}, Chat: up.URL + "/v1", Responses: up.URL + "/v1"}); err != nil {
		t.Fatal(err)
	}
	rec := httptest.NewRecorder()
	New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"probe-model","messages":[{"role":"user","content":"hello"}]}`)))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "probe ok") {
		t.Fatalf("valid in-flight reply lost: %d %s", rec.Code, rec.Body)
	}
	p, err := provider.Find("probe")
	if err != nil {
		t.Fatal(err)
	}
	if got, learned := p.LearnedAPI("probe-model"); learned {
		t.Fatalf("old wire success learned for new wire: %s", got)
	}
}

func TestAmbiguousProtocolProbeLocalTranslationFailure(t *testing.T) {
	for _, body := range []string{
		`{"model":"probe-model","input":[{"role":"user","content":[{"type":"input_image","file_id":"file_1"}]}]}`,
		`{"model":"probe-model","input":42}`,
	} {
		t.Run(body, func(t *testing.T) {
			t.Setenv("XDG_CONFIG_HOME", t.TempDir())
			t.Setenv("XDG_CACHE_HOME", t.TempDir())
			calls := &probeCounts{values: map[string]int{}}
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				io.Copy(io.Discard, r.Body)
				calls.hit(r.URL.Path)
				w.Header().Set("Content-Type", "application/json")
				w.Header().Set("X-Request-Id", "original-id")
				w.WriteHeader(404)
				io.WriteString(w, `{"error":{"message":"`+probeModelRefusal+`"}}`)
			}))
			t.Cleanup(up.Close)
			if err := provider.Save(provider.Provider{ID: "probe", Name: "Probe", Key: "k", Models: []string{"probe-model"}, Chat: up.URL + "/v1", Responses: up.URL + "/v1"}); err != nil {
				t.Fatal(err)
			}
			st := settings.Load()
			st.ModelImages = map[string]bool{"probe/probe-model": true}
			if err := settings.Save(st); err != nil {
				t.Fatal(err)
			}
			rec := httptest.NewRecorder()
			New().Handler().ServeHTTP(rec, httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body)))
			if rec.Code != 404 || rec.Header().Get("X-Request-Id") != "original-id" || !strings.Contains(rec.Body.String(), "does not exist") {
				t.Fatalf("original refusal lost: %d %v %s", rec.Code, rec.Header(), rec.Body)
			}
			if calls.count("/v1/responses") != 1 || calls.count("/v1/chat/completions") != 0 {
				t.Fatalf("untranslatable request retried upstream: %v", calls)
			}
		})
	}
}

func TestAmbiguousProtocolProbeContinuationStaysOnProtocol(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	calls := &probeCounts{values: map[string]int{}}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		calls.hit(r.URL.Path)
		if r.URL.Path == "/v1/responses" {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, probeGood)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		io.WriteString(w, `{"error":{"message":"model is not accessible via the /chat/completions endpoint, use /responses"}}`)
	}))
	t.Cleanup(up.Close)
	p := provider.Provider{ID: "probe", Name: "Probe", Key: "k", Chat: up.URL + "/v1", Responses: up.URL + "/v1"}
	probe := &protocolProbe{success: true, original: &http.Response{StatusCode: 404}, visited: map[provider.Protocol]bool{provider.Chat: true}}
	ctx := context.WithValue(context.Background(), protocolProbeKey{}, probe)
	res, actual, err := New().forwardTranslated(ctx, p, provider.Chat, &Request{Stream: true}, "probe-model", http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	if res.StatusCode != 400 || actual != provider.Chat || calls.count("/v1/chat/completions") != 1 || calls.count("/v1/responses") != 0 {
		t.Fatalf("continuation changed protocol after output: status=%d actual=%s calls=%v", res.StatusCode, actual, calls)
	}
}

func TestAmbiguousProtocolProbeCompletionKinds(t *testing.T) {
	responses := "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"usable partial\"}\n\n"
	chat := "data: {\"choices\":[{\"delta\":{\"content\":\"usable partial\"}}]}\n\n"
	messages := "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"usable partial\"}}\n\n"
	for _, tc := range []struct {
		name  string
		proto provider.Protocol
		body  string
		want  bool
	}{
		{"responses output limit", provider.Responses, responses + "event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n", true},
		{"responses content filter", provider.Responses, responses + "event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"content_filter\"}}}\n\n", false},
		{"responses unknown incomplete", provider.Responses, responses + "event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\"}}\n\n", false},
		{"responses failed", provider.Responses, responses + "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\"}}\n\n", false},
		{"responses transport truncation", provider.Responses, responses, false},
		{"chat output limit", provider.Chat, chat + "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n", true},
		{"chat content filter", provider.Chat, chat + "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"content_filter\"}]}\n\ndata: [DONE]\n\n", false},
		{"messages output limit", provider.Anthropic, messages + "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", true},
		{"messages refusal", provider.Anthropic, messages + "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\"}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res := &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(tc.body))}
			defer res.Body.Close()
			if got := validProbeReply(res, provider.Provider{}, tc.proto, &Request{}); got != tc.want {
				t.Fatalf("validProbeReply=%v want %v", got, tc.want)
			}
		})
	}
}
