package gateway

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"regexp"
	"slices"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

// protocolProbe is confined to one candidate and request. Ambiguous relay
// refusals aren't evidence against an endpoint shared by other credentials.
type protocolProbe struct {
	visited  map[provider.Protocol]bool
	original *http.Response
	body     []byte
	from     provider.Protocol
	own      string
	success  bool
}

type protocolProbeKey struct{}

func probeOf(ctx context.Context) *protocolProbe {
	p, _ := ctx.Value(protocolProbeKey{}).(*protocolProbe)
	return p
}

// newProtocolProbe limits discovery to a custom relay's single unrestricted
// key. Native or sealed histories must never be translated to discover an API.
func newProtocolProbe(p provider.Provider, model string, body []byte) *protocolProbe {
	if p.Preset != "" || p.Account != nil || p.Key == "" || len(p.Keys) != 0 || p.KeyProtocol != "" || p.PassesErrors() {
		return nil
	}
	if _, explicit := p.ModelAPI(model); explicit {
		return nil
	}
	for _, field := range []string{`"compaction_trigger"`, `"compaction"`, `"encrypted_content"`, `"encrypted_function_args"`, `"signature"`, `"previous_response_id"`} {
		if bytes.Contains(body, []byte(field)) {
			return nil
		}
	}
	return &protocolProbe{visited: map[provider.Protocol]bool{}, own: p.OwnModelID(model)}
}

var ambiguousModelProtocol = regexp.MustCompile(`(?i)\bmodel\b[^\r\n]{0,256}\bdoes not exist or you do not have access to it\b`)
var probeCredentialPolicy = regexp.MustCompile(`(?i)unauthori[sz]ed|authentication|invalid.{0,24}(key|token|credential)|api.?key|permission denied|forbidden|policy|content.filter|safety|blocked`)

func ambiguousProtocolRefusal(status int, body []byte) bool {
	if status != 400 && status != 404 {
		return false
	}
	body = wordsOf(body)
	if quotaWords.Match(body) || creditWords.Match(body) || refusedWords.Match(body) || regionWords.Match(body) || probeCredentialPolicy.Match(body) || zdrRefused(status, body) {
		return false
	}
	words := strings.ToLower(string(body))
	return ambiguousModelProtocol.Match(body) ||
		strings.Contains(words, "url error, please check url！") || strings.Contains(words, "url error, please check url!")
}

func (p *protocolProbe) save(res *http.Response, proto provider.Protocol, body []byte) {
	if p.original != nil {
		return
	}
	original := *res
	original.Header = res.Header.Clone()
	p.original, p.body, p.from = &original, bytes.Clone(body), proto
	p.visited[proto] = true
}

func (p *protocolProbe) next(s *Server, upstream provider.Provider, model string) (provider.Protocol, bool) {
	for _, proto := range s.usable(upstream, model) {
		if !p.visited[proto] && slices.Contains([]provider.Protocol{provider.Chat, provider.Responses, provider.Anthropic, provider.Gemini}, proto) {
			return proto, true
		}
	}
	return "", false
}

func (p *protocolProbe) restore(ctx context.Context, upstream provider.Provider) (*http.Response, provider.Protocol, error) {
	p.original.Body = io.NopCloser(bytes.NewReader(p.body))
	heardReply(ctx, upstream, p.original)
	return p.original, p.from, nil
}

// probeWriter suppresses local translation errors until a fully validated
// upstream reply exists. From that point it preserves normal streaming.
type probeWriter struct {
	http.ResponseWriter
	probe *protocolProbe
}

func (w *probeWriter) WriteHeader(status int) {
	if w.probe.success {
		w.ResponseWriter.WriteHeader(status)
	}
}

func (w *probeWriter) Write(body []byte) (int, error) {
	if w.probe.success {
		return w.ResponseWriter.Write(body)
	}
	return len(body), nil
}

func (w *probeWriter) Flush() {
	if w.probe.success {
		if f, ok := w.ResponseWriter.(http.Flusher); ok {
			f.Flush()
		}
	}
}

func (s *Server) translateProbed(w http.ResponseWriter, r *http.Request, p provider.Provider, from, to provider.Protocol, model string, body []byte, u *Usage, probe *protocolProbe) (int, string) {
	head, before := w.Header().Clone(), *u
	status, msg := s.translate(&probeWriter{ResponseWriter: w, probe: probe}, r, p, from, to, model, body, u)
	if probe.success {
		return status, msg
	}
	// Parsing, building or an unportable file may fail before forwarding.
	// Those local errors must not replace the refusal that started discovery.
	clear(w.Header())
	for k, values := range head {
		w.Header()[k] = values
	}
	*u = before
	res, actual, _ := probe.restore(r.Context(), p)
	defer res.Body.Close()
	u.RequestID = requestID(res.Header)
	u.ErrType = provider.ErrorType(probe.body)
	msg = explained(p, model, res, probe.body)
	keepRetry(w.Header(), res.Header, probe.body)
	return writeVendorErrorFrom(w, from, actual, p, res, msg, probe.body), msg
}

// validProbeReply buffers only a discovery reply, before any output escapes.
// A status alone, an empty stream, an error or a truncated generation proves
// nothing. Normal replies retain their existing streaming behavior.
func validProbeReply(res *http.Response, upstream provider.Provider, proto provider.Protocol, req *Request) bool {
	const limit = 4 << 20
	body, err := io.ReadAll(io.LimitReader(res.Body, limit+1))
	res.Body.Close()
	res.Body = io.NopCloser(bytes.NewReader(body))
	if err != nil || len(body) > limit {
		return false
	}
	rd, stream := eventStream(res)
	if !stream {
		return false
	}
	dec := decoderFor(upstream, proto, req.Tools)
	content, ended, failed := false, false, false
	err = readSSE(rd, func(_, data string) error {
		if _, bad := carriedError(data); bad {
			failed = true
		}
		// A named output limit is a legitimate completed generation. An
		// unknown incomplete reason, filtering or policy rejection is not
		// evidence that this credential serves the requested model.
		if proto == provider.Responses && (eventType(data) == "response.incomplete" || gjson.Get(data, "response.status").Str == "incomplete") && gjson.Get(data, "response.incomplete_details.reason").Str != "max_output_tokens" {
			failed = true
		}
		ended = ended || replyEnds(proto, data, upstreamStop([]byte(data)))
		return dec(data, func(e Event) {
			switch e.Kind {
			case KText, KImage:
				content = content || strings.TrimSpace(e.Text) != ""
			case KToolStart:
				content = content || e.Name != ""
			case KStop:
				failed = failed || e.Stop == "filter"
			case KError:
				failed = true
			}
		})
	})
	res.Body = io.NopCloser(bytes.NewReader(body))
	return err == nil && !failed && content && ended
}
