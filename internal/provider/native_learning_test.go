package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/catalog"
	"github.com/yetone/magpie/internal/settings"
)

func TestNativeLearningPerModelPreservesBase(t *testing.T) {
	detectHome(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			io.WriteString(w, `{"data":[{"id":"a","native_endpoints":["/v1/messages"]},{"id":"b","native_endpoints":["/v1/messages"]}]}`)
			return
		}
		var body struct {
			Model string `json:"model"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if r.URL.Path == "/v1/messages" && body.Model == "a" {
			io.WriteString(w, nativeMessagesAnswer)
			return
		}
		if r.URL.Path == "/v1/chat/completions" && body.Model == "b" {
			io.WriteString(w, `{"choices":[{"message":{"content":"OK"}}]}`)
			return
		}
		http.Error(w, "not served", 404)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"a", "b"}, Proxy: "direct"}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	q := mustNativeProvider(t, p.ID)
	if q.Chat != p.Chat || q.BaseAPI != p.BaseAPI || q.Anthropic != srv.URL {
		t.Fatalf("lost original configuration: %+v", q)
	}
	if q.Native("a") != Anthropic || q.Native("b") != Chat || q.NoAPIFor("b") {
		t.Fatalf("models did not keep their own verified APIs: a=%s b=%s barred=%v", q.Native("a"), q.Native("b"), q.NoAPIFor("b"))
	}
	if len(settings.Load().ModelAPIs) != 0 {
		t.Fatal("automatic discovery wrote user API preferences")
	}
}

func TestNativeLearningCacheScopeAndSoftExpiry(t *testing.T) {
	detectHome(t)
	p := Provider{ID: "relay", Key: "first", Chat: "http://relay.invalid/v1", Anthropic: "http://relay.invalid", BaseAPI: "chat"}
	if !p.RememberAPI("a", Anthropic) {
		t.Fatal("success not remembered")
	}
	if pr, ok := p.LearnedAPI("a"); !ok || pr != Anthropic {
		t.Fatal("success not available")
	}
	if _, ok := p.LearnedAPI("b"); ok {
		t.Fatal("one model taught another model")
	}
	for _, change := range []func(*Provider){
		func(q *Provider) { q.Key = "second" }, func(q *Provider) { q.Anthropic = "http://other.invalid" },
		func(q *Provider) { q.Headers = map[string]string{"X-Tenant": "other"} }, func(q *Provider) { q.ExtraBody = ExtraBody{"tenant": json.RawMessage(`"other"`)} },
	} {
		q := p
		change(&q)
		if _, ok := q.LearnedAPI("a"); ok {
			t.Fatal("success crossed credentials or configuration")
		}
	}
	if p.WithKey(KeyAccount{Key: "second"}).RememberAPI("a", Chat) == false {
		t.Fatal("second key could not learn")
	}
	if pr, _ := p.LearnedAPI("a"); pr != Anthropic {
		t.Fatal("second key polluted first key")
	}
	bound := p.WithKey(KeyAccount{Key: "first", Protocol: Chat})
	if bound.RememberAPI("a", Anthropic) {
		t.Fatal("bound key learned a removed endpoint")
	}
	key, _ := p.learningKey("a")
	learnedAPIs.Lock()
	v := learnedAPIs.entries[key]
	v.Until = time.Now().Add(-time.Hour)
	learnedAPIs.entries[key] = v
	learnedAPIs.Unlock()
	if pr, ok := p.LearnedAPI("a"); !ok || pr != Anthropic {
		t.Fatal("soft expiry discarded successful evidence")
	}
	if !p.RememberAPI("a", Chat) {
		t.Fatal("could not refresh success")
	}
	s := settings.Load()
	s.ModelWires = map[string]string{"relay/a": "changed-upstream"}
	if err := settings.Save(s); err != nil {
		t.Fatal(err)
	}
	if _, ok := p.LearnedAPI("a"); ok {
		t.Fatal("upstream wire mapping change retained stale evidence")
	}
}

func TestNativeLearningUnknownDoesNotAdvertiseAllAPIs(t *testing.T) {
	detectHome(t)
	p := Provider{ID: "relay", Key: "fake", Chat: "http://relay.invalid/v1", Anthropic: "http://relay.invalid"}
	if got := NativeEndpoints(Entry{Provider: p, Model: "unknown"}); len(got) != 0 {
		t.Fatalf("unknown advertised native endpoints: %v", got)
	}
	if !p.RememberAPI("known", Chat) {
		t.Fatal("could not record configured API success")
	}
	if got := NativeEndpoints(Entry{Provider: p, Model: "known"}); len(got) != 1 || got[0] != "/v1/chat/completions" {
		t.Fatalf("known endpoints: %v", got)
	}
}

func TestNativeLearningInferredEndpointNeedsEachModelsEvidence(t *testing.T) {
	detectHome(t)
	p := Provider{ID: "relay", Key: "fake", Chat: "http://relay.invalid/v1", Anthropic: "http://relay.invalid", NativeInferred: map[Protocol]string{Anthropic: "http://relay.invalid"}}
	if err := catalog.SaveLive(p.ID, p.Chat, []catalog.Model{{ID: "a", APIs: []string{"anthropic"}}, {ID: "b", APIs: []string{"anthropic"}}}); err != nil {
		t.Fatal(err)
	}
	if !p.RememberAPI("a", Anthropic) {
		t.Fatal("could not learn model a")
	}
	if p.NoAPIFor("a") || !p.NoAPIFor("b") || len(NativeEndpoints(Entry{Provider: p, Model: "b"})) != 0 {
		t.Fatal("one model's inferred endpoint opened another model")
	}
	forgetLearningMemory()
	if p.NoAPIFor("a") {
		t.Fatal("restart lost persisted successful evidence")
	}
	p.Anthropic = "http://manual.invalid"
	if p.NoAPIFor("a") {
		t.Fatal("editing the inferred URL did not restore explicit endpoint semantics")
	}
}

// A list with more declared models than one fetch may verify verifies the
// first 24 and leaves the rest unknown, not unsupported. Models the list
// says nothing about are never verified: a vendor that lists hundreds
// without a per-model declaration (OpenRouter's) would otherwise pay a
// generation for each on every refresh.
func TestNativeLearningDeclaredModelsHaveSharedBudget(t *testing.T) {
	detectHome(t)
	var posts atomic.Int32
	var models []string
	for i := 0; i < 40; i++ {
		models = append(models, fmt.Sprintf(`{"id":"model-%d","native_endpoints":["/v1/messages"]}`, i))
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			io.WriteString(w, `{"data":[`+strings.Join(models, ",")+`]}`)
			return
		}
		posts.Add(1)
		io.WriteString(w, nativeMessagesAnswer)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake", Chat: srv.URL + "/v1", Responses: srv.URL + "/v1", BaseAPI: "chat"}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	if posts.Load() != 24 {
		t.Fatalf("generation count = %d, want shared budget 24", posts.Load())
	}
	q := mustNativeProvider(t, p.ID)
	if _, ok := q.LearnedAPI("model-24"); ok {
		t.Fatal("an unprobed model gained success evidence")
	}
}

// A list that says nothing about its models is refreshed without a
// generation request, however many endpoints the provider has.
func TestNativeLearningUndeclaredModelsCostNoGeneration(t *testing.T) {
	detectHome(t)
	var posts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			io.WriteString(w, `{"data":[{"id":"a"},{"id":"b"}]}`)
			return
		}
		posts.Add(1)
		io.WriteString(w, nativeMessagesAnswer)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake", Chat: srv.URL + "/v1", Responses: srv.URL + "/v1", Anthropic: srv.URL, BaseAPI: "chat"}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	if posts.Load() != 0 {
		t.Fatalf("a list with no declaration paid %d generations", posts.Load())
	}
}

// A provider whose list comes from a URL the user gave is listed there and
// nowhere else, and nothing of it is generated for.
func TestNativeLearningListURLAloneIsAsked(t *testing.T) {
	detectHome(t)
	var lists, posts atomic.Int32
	list := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		lists.Add(1)
		io.WriteString(w, `{"data":[{"id":"only","native_endpoints":["/v1/messages"]}]}`)
	}))
	defer list.Close()
	base := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		posts.Add(1)
		io.WriteString(w, nativeMessagesAnswer)
	}))
	defer base.Close()
	p := Provider{ID: "listed", Name: "Listed", Key: "fake", Chat: base.URL + "/v1", Anthropic: base.URL, BaseAPI: "chat", ModelsURL: list.URL + "/v1/models"}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	q, err := Find(p.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	if lists.Load() != 1 || posts.Load() != 0 {
		t.Fatalf("asked its list %d times and a base %d times", lists.Load(), posts.Load())
	}
}

// A refresh magpie schedules itself takes what the last one learned without
// paying again: only a fetch the user asked for verifies.
func TestNativeLearningScheduledRefreshPaysNoGeneration(t *testing.T) {
	detectHome(t)
	var posts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			io.WriteString(w, nativeMessagesList)
			return
		}
		posts.Add(1)
		io.WriteString(w, nativeMessagesAnswer)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"zcode/GLM-5.3-Flash"}}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	if _, err := p.Fetch(WithBackgroundList(context.Background())); err != nil {
		t.Fatal(err)
	}
	if posts.Load() != 0 {
		t.Fatalf("a scheduled refresh paid %d generations", posts.Load())
	}
	// and the user's own Fetch still repairs the provider
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	if posts.Load() == 0 {
		t.Fatal("a fetch the user asked for verified nothing")
	}
}

func TestNativeLearningFailureCoolsOnlyThisModel(t *testing.T) {
	detectHome(t)
	var posts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			io.WriteString(w, nativeMessagesList)
			return
		}
		posts.Add(1)
		http.Error(w, "not served", 404)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"zcode/GLM-5.3-Flash"}}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	for i := 0; i < 2; i++ {
		if _, err := p.Fetch(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if posts.Load() != 2 {
		t.Fatalf("failure was immediately reprobed: %d generations", posts.Load())
	}
	if _, ok := p.LearnedAPI(p.Models[0]); ok {
		t.Fatal("failure became success evidence")
	}
	key, _ := p.learningKey(p.Models[0])
	learnedAPIs.Lock()
	v := learnedAPIs.entries[key]
	v.Until = time.Now().Add(-time.Hour)
	learnedAPIs.entries[key] = v
	learnedAPIs.Unlock()
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	if posts.Load() != 4 {
		t.Fatal("failure cooldown became permanent")
	}
}

func TestNativeLearningDoesNotProbeUnambiguousUnknownSingleAPI(t *testing.T) {
	detectHome(t)
	var posts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" {
			posts.Add(1)
			http.Error(w, "unexpected generation", 400)
			return
		}
		io.WriteString(w, `{"data":[{"id":"a"}]}`)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake", Chat: srv.URL + "/v1", BaseAPI: "chat"}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	if posts.Load() != 0 {
		t.Fatal("an ordinary one-API list fetch generated a paid request")
	}
}

func TestNativeLearningChangedOnlyOnNewPreference(t *testing.T) {
	detectHome(t)
	p := Provider{ID: "relay", Key: "fake", Chat: "http://relay.invalid/v1", Anthropic: "http://relay.invalid"}
	before := StampNow()
	p.RememberAPI("a", Chat)
	after := StampNow()
	if before == after {
		t.Fatal("successful API discovery did not invalidate catalog readers")
	}
	p.RememberAPI("a", Chat)
	if after != StampNow() {
		t.Fatal("renewing the same success repeatedly invalidated catalog readers")
	}
	p.RememberAPI("a", Anthropic)
	if after == StampNow() {
		t.Fatal("changed API preference did not invalidate catalog readers")
	}
}

func TestNativeLearningKeepsMultipleAndProtocolBoundKeys(t *testing.T) {
	for _, protocolBound := range []bool{false, true} {
		t.Run(fmt.Sprint(protocolBound), func(t *testing.T) {
			detectHome(t)
			var posts atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "POST" {
					posts.Add(1)
					io.WriteString(w, nativeMessagesAnswer)
					return
				}
				io.WriteString(w, nativeMessagesList)
			}))
			defer srv.Close()
			p := Provider{ID: "relay", Name: "Relay", Key: "first", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"zcode/GLM-5.3-Flash"}}
			if protocolBound {
				p.KeyProtocol = Chat
			} else {
				p.Keys = []KeyAccount{{Key: "second"}}
			}
			if err := Save(p); err != nil {
				t.Fatal(err)
			}
			p = mustNativeProvider(t, p.ID)
			if _, err := p.Fetch(context.Background()); err != nil {
				t.Fatal(err)
			}
			q := mustNativeProvider(t, p.ID)
			if posts.Load() != 0 || q.Chat != p.Chat || q.Anthropic != "" || q.BaseAPI != p.BaseAPI {
				t.Fatal("multi-key or protocol-bound provider was automatically probed or changed")
			}
		})
	}
}

// forgetLearningMemory is a restart: nothing of what this process learned is
// in memory any more, and the file is read again as a new magpie reads it.
func forgetLearningMemory() {
	learnedAPIs.Lock()
	clear(learnedAPIs.entries)
	clear(learnedAPIs.scopes)
	clear(learnedAPIs.loaded)
	clear(learnedAPIs.unreadable)
	learnedAPIs.Unlock()
	learnedReady.Store(false)
	learnedCount.Store(0)
}

func TestNativeLearningSingleManualUnknownAPIIsNative(t *testing.T) {
	detectHome(t)
	p := Provider{ID: "relay", Key: "fake", Anthropic: "http://relay.invalid", BaseAPI: "anthropic"}
	got := NativeEndpoints(Entry{Provider: p, Model: "unknown"})
	if len(got) != 1 || got[0] != "/v1/messages" {
		t.Fatalf("single explicit Messages base lost its native endpoint: %v", got)
	}
}

func TestNativeLearningSoftExpiryAndFailedRefreshKeepRouting(t *testing.T) {
	detectHome(t)
	var failing atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			io.WriteString(w, nativeMessagesList)
			return
		}
		if failing.Load() {
			http.Error(w, "temporarily down", 503)
			return
		}
		io.WriteString(w, nativeMessagesAnswer)
	}))
	defer srv.Close()
	p := Provider{ID: "relay", Name: "Relay", Key: "fake", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"zcode/GLM-5.3-Flash"}}
	if err := Save(p); err != nil {
		t.Fatal(err)
	}
	p = mustNativeProvider(t, p.ID)
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	q := mustNativeProvider(t, p.ID)
	model := q.Models[0]
	key, _ := q.learningKey(model)
	learnedAPIs.Lock()
	v := learnedAPIs.entries[key]
	v.Until = time.Now().Add(-time.Hour)
	learnedAPIs.entries[key] = v
	learnedAPIs.Unlock()
	if q.NoAPIFor(model) || q.Native(model) != Anthropic {
		t.Fatal("soft TTL removed working route")
	}
	failing.Store(true)
	if _, err := q.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	if q.NoAPIFor(model) || q.Native(model) != Anthropic {
		t.Fatal("failed refresh removed last successful route")
	}
	forgetLearningMemory()
	if q.NoAPIFor(model) || q.Native(model) != Anthropic {
		t.Fatal("restart removed last successful route")
	}
}

func TestNativeLearningManualDeclarationNeedsNoRepair(t *testing.T) {
	for _, proto := range []Protocol{Chat, Anthropic} {
		t.Run(string(proto), func(t *testing.T) {
			detectHome(t)
			var posts atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "POST" {
					posts.Add(1)
					http.Error(w, "unexpected generation", 400)
					return
				}
				fmt.Fprintf(w, `{"data":[{"id":"a","native_endpoints":[%q]}]}`, endpointOf[proto])
			}))
			defer srv.Close()
			p := Provider{ID: "relay", Name: "Relay", Key: "fake", BaseAPI: string(proto), Models: []string{"a"}}
			if proto == Chat {
				p.Chat = srv.URL + "/v1"
			} else {
				p.Anthropic = srv.URL
			}
			if err := Save(p); err != nil {
				t.Fatal(err)
			}
			p = mustNativeProvider(t, p.ID)
			if _, err := p.Fetch(context.Background()); err != nil {
				t.Fatal(err)
			}
			if posts.Load() != 0 || p.Native("a") != proto {
				t.Fatal("a valid manual declaration caused paid repair or lost its native preference")
			}
		})
	}
}

func TestNativeLearningDoesNotAttributeOldReplyToChangedMapping(t *testing.T) {
	for _, localID := range []bool{false, true} {
		t.Run(fmt.Sprint(localID), func(t *testing.T) {
			detectHome(t)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == "GET" {
					io.WriteString(w, nativeMessagesList)
					return
				}
				s := settings.Load()
				if localID {
					s.ModelIDs = map[string]string{"relay/zcode/GLM-5.3-Flash": "different-model"}
				} else {
					s.ModelWires = map[string]string{"relay/zcode/GLM-5.3-Flash": "different-model"}
				}
				if err := settings.Save(s); err != nil {
					t.Error(err)
				}
				io.WriteString(w, nativeMessagesAnswer)
			}))
			defer srv.Close()
			p := Provider{ID: "relay", Name: "Relay", Key: "fake", Chat: srv.URL + "/v1", BaseAPI: "chat", Models: []string{"zcode/GLM-5.3-Flash"}}
			if err := Save(p); err != nil {
				t.Fatal(err)
			}
			p = mustNativeProvider(t, p.ID)
			if _, err := p.Fetch(context.Background()); err != nil {
				t.Fatal(err)
			}
			q := mustNativeProvider(t, p.ID)
			if q.Anthropic != "" {
				t.Fatal("old reply changed configuration after the model mapping was edited")
			}
			if _, ok := q.LearnedAPI(p.Models[0]); ok {
				t.Fatal("old reply was remembered under a new model mapping")
			}
		})
	}
}

func TestNativeLearningSingleAPIExplicitOverride(t *testing.T) {
	detectHome(t)
	p := Provider{ID: "relay", Chat: "http://relay.invalid/v1"}
	s := settings.Load()
	s.ModelAPIs = map[string]string{"relay/a": "chat"}
	if err := settings.Save(s); err != nil {
		t.Fatal(err)
	}
	if pr, ok := p.ModelAPI("a"); !ok || pr != Chat {
		t.Fatal("single API explicit override was ignored")
	}
}
