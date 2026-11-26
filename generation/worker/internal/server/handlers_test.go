package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/psyb0t/docker-imagies/generation/worker/internal/config"
)

func TestResolveModelKey(t *testing.T) {
	srv := &Server{
		modelPrefix: "imagies-",
		registry: &config.Registry{
			Models: map[string]config.ModelConfig{"sd-turbo": {}},
		},
	}

	for _, id := range []string{"sd-turbo", "imagies-sd-turbo", "openai/imagies-sd-turbo"} {
		t.Run(id, func(t *testing.T) {
			if key, err := srv.resolveModelKey(id); err != nil || key != "sd-turbo" {
				t.Fatalf("resolveModelKey(%q) = %q, %v", id, key, err)
			}
		})
	}

	if _, err := srv.resolveModelKey("other-engine-sd-turbo"); err == nil {
		t.Fatal("accepted a model from another engine")
	}
}

func TestUnloadBusy(t *testing.T) {
	for _, test := range []struct {
		name string
		path string
		want int
	}{
		{"legacy", "/sdcpp/v1/unload", http.StatusOK},
		{"strict", "/sdcpp/v1/unload?fail_if_busy=true", http.StatusConflict},
	} {
		t.Run(test.name, func(t *testing.T) {
			srv := &Server{generating: true, loaded: true}
			response := httptest.NewRecorder()
			NewUnloadHandler(srv).ServeHTTP(response, httptest.NewRequest(http.MethodPost, test.path, nil))
			if response.Code != test.want || !srv.loaded {
				t.Fatalf("unload returned %d, resident model loaded: %t", response.Code, srv.loaded)
			}
		})
	}
}
