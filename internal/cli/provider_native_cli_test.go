package cli

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

func TestProviderAnnounceUsesCorrectedNativeAPI(t *testing.T) {
	groupsHome(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method + " " + r.URL.Path {
		case "GET /v1/models":
			io.WriteString(w, `{"data":[{"id":"zcode/GLM-5.3-Flash","native_endpoints":["/v1/messages"]}]}`)
		case "POST /v1/messages":
			io.WriteString(w, `{"type":"message","content":[{"type":"text","text":"OK"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	if err := provider.Save(provider.Provider{ID: "relay", Name: "Relay", Key: "fake-key", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"zcode/GLM-5.3-Flash"}}); err != nil {
		t.Fatal(err)
	}
	out, err := stdoutOf(t, func() error { return announce("relay") })
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "no models exposed") || !strings.Contains(out, "1 models in the catalog") {
		t.Fatalf("announce kept the old API's empty catalog: %s", out)
	}
}
