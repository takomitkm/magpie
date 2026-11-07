package provider

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"os"
	"testing"

	"github.com/yetone/magpie/internal/settings"
)

func TestNativeLearningReaderEmptyHasNoPerModelAllocations(t *testing.T) {
	detectHome(t)
	p := Provider{ID: "empty-relay", Key: "fake", Chat: "http://relay.invalid/v1"}
	lookup := p.LearnedAPIReader()
	n := testing.AllocsPerRun(10, func() {
		for range 2000 {
			lookup("unknown")
		}
	})
	if n != 0 {
		t.Fatalf("an empty owner's 2000 model lookups allocated %v times", n)
	}
}

func TestNativeLearningReaderKeepsPersistedFingerprint(t *testing.T) {
	detectHome(t)
	p := Provider{ID: "relay", Key: "fake", Chat: "http://relay.invalid/v1", BaseAPI: "chat", Headers: map[string]string{"X-Tenant": "one"}}
	s := settings.Load()
	s.ModelIDs = map[string]string{"relay/local": "upstream"}
	s.ModelWires = map[string]string{"relay/local": "wire-model"}
	if err := settings.Save(s); err != nil {
		t.Fatal(err)
	}
	// This is the persisted key format before batch readers existed.
	config := Provider{ID: p.ID, Key: p.Key, KeyProtocol: p.KeyProtocol, KeyPlan: p.KeyPlan,
		Chat: p.Chat, Responses: p.Responses, Anthropic: p.Anthropic, Gemini: p.Gemini,
		BaseAPI: p.BaseAPI, NativeInferred: p.NativeInferred, Headers: p.Headers,
		ExtraBody: p.ExtraBody, PassHeaders: p.PassHeaders, AuthHeader: p.AuthHeader, Proxy: p.ProxyChoice(),
		InsecureTLS: p.InsecureTLS, HTTP1: p.HTTP1, NativeCompaction: p.NativeCompaction}
	b, err := json.Marshal(struct {
		Path        string
		Config      Provider
		Model, Wire string
	}{Path(), config, "upstream", "wire-model"})
	if err != nil {
		t.Fatal(err)
	}
	key, ok := p.learningKey("local")
	if !ok || key != sha256.Sum256(b) {
		t.Fatal("batch fingerprint invalidated previously persisted success")
	}
	if !p.RememberAPI("local", Chat) {
		t.Fatal("could not record success")
	}
	if b, err := os.ReadFile(learnedAPIPath()); err != nil || !bytes.Contains(b, []byte("chat")) {
		t.Fatalf("success was not written to %s: %v %s", learnedAPIPath(), err, b)
	}
	forgetLearningMemory()
	if proto, ok := p.LearnedAPIReader()("local"); !ok || proto != Chat {
		t.Fatal("batch reader did not restore persisted model success")
	}
}

func TestNativeLearningReaderUsesOneMappingSnapshot(t *testing.T) {
	detectHome(t)
	p := Provider{ID: "relay", Key: "first", Chat: "http://relay.invalid/v1"}
	if !p.RememberAPI("a", Chat) {
		t.Fatal("could not record success")
	}
	lookup := p.LearnedAPIReader()
	s := settings.Load()
	s.ModelWires = map[string]string{"relay/a": "changed-upstream"}
	if err := settings.Save(s); err != nil {
		t.Fatal(err)
	}
	if proto, ok := lookup("a"); !ok || proto != Chat {
		t.Fatal("a batch changed its model identity halfway through")
	}
	if _, ok := p.LearnedAPIReader()("a"); ok {
		t.Fatal("a new batch reused success for the old wire mapping")
	}
	if _, ok := p.WithKey(KeyAccount{Key: "second"}).LearnedAPIReader()("a"); ok {
		t.Fatal("batch evidence crossed credentials")
	}
}
