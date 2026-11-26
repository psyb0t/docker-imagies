package process

import (
	"testing"
	"time"
)

func TestStopDoesNotWaitForExitCallback(t *testing.T) {
	manager := NewManager("/bin/true", "127.0.0.1:1234", nil, false)
	entered := make(chan struct{})
	release := make(chan struct{})
	defer close(release)
	manager.SetOnExit(func() {
		close(entered)
		<-release
	})
	if err := manager.Start(nil); err != nil {
		t.Fatalf("start: %v", err)
	}
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("exit callback did not run")
	}
	stopped := make(chan struct{})
	go func() {
		manager.Stop()
		close(stopped)
	}()
	select {
	case <-stopped:
	case <-time.After(2 * time.Second):
		t.Fatal("stop blocked behind the exit callback")
	}
}
