package gui

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// The editor doesn't send inference provenance. Saving a name or copying a
// provider must not turn an inferred endpoint into a manually configured one.
func TestProviderEditorKeepsInferredEndpoints(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("USERPROFILE", os.Getenv("HOME"))
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	mux := http.NewServeMux()
	providerRoutes(mux, nil)
	const chat = "http://127.0.0.1:1/v1"
	const messages = "http://127.0.0.1:1"
	seed := func(t *testing.T) {
		t.Helper()
		if err := provider.Save(provider.Provider{
			ID: "inferred", Name: "Inferred", Key: "fake", Chat: chat,
			Anthropic: messages, BaseAPI: "chat",
			NativeInferred: map[provider.Protocol]string{provider.Anthropic: messages},
		}); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct {
		name, base string
		copy       bool
	}{
		{"edit unchanged base", messages, false},
		{"edit changed base", "http://127.0.0.1:1/other", false},
		{"copy unchanged base", messages, true},
		{"copy changed base", "http://127.0.0.1:1/other", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			seed(t)
			body := map[string]any{"id": "inferred", "from": "inferred", "name": "Edited", "chat": chat, "anthropic": tc.base, "baseAPI": "chat"}
			id := "inferred"
			if tc.copy {
				id = "inferred-copy"
				body["id"], body["name"], body["new"], body["copyOf"] = id, "Copied", true, "inferred"
				delete(body, "from")
			}
			raw, err := json.Marshal(body)
			if err != nil {
				t.Fatal(err)
			}
			w := httptest.NewRecorder()
			mux.ServeHTTP(w, httptest.NewRequest("POST", "/api/provider/save", strings.NewReader(string(raw))))
			if w.Code != http.StatusOK {
				t.Fatalf("save: %d %s", w.Code, w.Body)
			}
			p, err := provider.Find(id)
			if err != nil {
				t.Fatal(err)
			}
			want := ""
			if tc.base == messages {
				want = messages
			}
			if got := p.NativeInferred[provider.Anthropic]; got != want {
				t.Fatalf("inference provenance = %q, want %q (base %q)", got, want, p.Anthropic)
			}
			if tc.copy {
				if err := provider.Delete(id); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
