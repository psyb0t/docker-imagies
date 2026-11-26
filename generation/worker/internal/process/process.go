package process

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/psyb0t/ctxerrors"
)

// Manager handles the sd-server subprocess lifecycle.
type Manager struct {
	sdServerPath string
	listenAddr   string
	extraArgs    []string
	verbose      bool
	onExit       func()

	mu      sync.Mutex
	current *childProcess
}

type childProcess struct {
	command  *exec.Cmd
	done     chan struct{}
	stopping atomic.Bool
}

func NewManager(sdServerPath, listenAddr string, extraArgs []string, verbose bool) *Manager {
	return &Manager{
		sdServerPath: sdServerPath,
		listenAddr:   listenAddr,
		extraArgs:    extraArgs,
		verbose:      verbose,
	}
}

// SetOnExit sets a callback invoked when sd-server exits unexpectedly.
func (m *Manager) SetOnExit(fn func()) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onExit = fn
}

func (m *Manager) Start(modelArgs []string) error {
	args := make([]string, 0, len(modelArgs)+len(m.extraArgs)+4)
	args = append(args, modelArgs...)
	args = append(args, "--listen-ip", listenIP(m.listenAddr))
	args = append(args, "--listen-port", listenPort(m.listenAddr))
	args = append(args, m.extraArgs...)
	if m.verbose {
		args = append(args, "-v")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	cmd := exec.Command(m.sdServerPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	slog.Info("starting sd-server", "path", m.sdServerPath, "args", args)

	if err := cmd.Start(); err != nil {
		return ctxerrors.Wrap(err, "start sd-server")
	}

	child := &childProcess{command: cmd, done: make(chan struct{})}
	m.current = child
	go m.wait(child)
	return nil
}

func (m *Manager) wait(child *childProcess) {
	if err := child.command.Wait(); err != nil {
		slog.Warn("sd-server exited", "error", err)
	} else {
		slog.Info("sd-server exited cleanly")
	}
	close(child.done)

	m.mu.Lock()
	callback := m.onExit
	current := m.current == child
	m.mu.Unlock()
	if current && !child.stopping.Load() && callback != nil {
		callback()
	}
}

func (m *Manager) Stop() {
	m.mu.Lock()
	child := m.current
	if child == nil {
		m.mu.Unlock()
		return
	}
	firstStop := !child.stopping.Swap(true)
	m.mu.Unlock()

	if firstStop {
		m.stopChild(child)
	} else {
		<-child.done
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.current == child {
		m.current = nil
	}
}

func (m *Manager) stopChild(child *childProcess) {
	select {
	case <-child.done:
		return
	default:
	}
	pid := child.command.Process.Pid
	slog.Info("stopping sd-server", "pid", pid)
	_ = child.command.Process.Signal(syscall.SIGTERM)
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	select {
	case <-child.done:
		slog.Info("sd-server stopped gracefully")
	case <-deadline.C:
		slog.Warn("sd-server kill timeout, sending SIGKILL", "pid", pid)
		_ = syscall.Kill(-pid, syscall.SIGKILL)
		<-child.done
	}
}

func (m *Manager) Running() bool {
	return m.PID() != 0
}

func (m *Manager) child() *childProcess {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.current
}

// WaitReady polls sd-server's /v1/models until it responds 200,
// the process exits, or ctx expires.
func (m *Manager) WaitReady(ctx context.Context) error {
	child := m.child()
	if child == nil {
		return ErrNotReady
	}
	url := fmt.Sprintf("http://%s/v1/models", m.listenAddr)
	client := &http.Client{Timeout: 2 * time.Second}
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctxerrors.Wrap(ErrNotReady, ctx.Err().Error())
		case <-child.done:
			return ctxerrors.New("sd-server exited during startup")
		case <-ticker.C:
			resp, err := client.Get(url)
			if err == nil {
				_ = resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					slog.Info("sd-server ready")
					return nil
				}
			}
		}
	}
}

// PID returns the sd-server process ID, or 0 if not running.
func (m *Manager) PID() int {
	child := m.child()
	if child == nil {
		return 0
	}
	select {
	case <-child.done:
		return 0
	default:
		return child.command.Process.Pid
	}
}

// ListenAddr returns the address sd-server listens on.
func (m *Manager) ListenAddr() string {
	return m.listenAddr
}

func listenIP(addr string) string {
	for i := len(addr) - 1; i >= 0; i-- {
		if addr[i] == ':' {
			return addr[:i]
		}
	}
	return addr
}

func listenPort(addr string) string {
	for i := len(addr) - 1; i >= 0; i-- {
		if addr[i] == ':' {
			return addr[i+1:]
		}
	}
	return addr
}
