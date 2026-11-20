// Package watcher closes the async loop: it periodically checks whether the
// mining engine has published analytics for dispatched jobs and flips them
// to COMPLETED, so clients polling GET /api/v1/jobs/:id see the terminal
// state without the gateway ever blocking on the Python pipeline.
package watcher

import (
	"log"
	"time"

	"spatiotemporal/gateway/internal/queue"
	"spatiotemporal/gateway/internal/storage"
)

// Watcher sweeps dispatched jobs against the analytics directory.
type Watcher struct {
	queue    *queue.InMemoryQueue
	buffer   *storage.FileBuffer
	interval time.Duration
	quit     chan struct{}
	done     chan struct{}
}

// New builds a watcher; call Start to begin sweeping.
func New(q *queue.InMemoryQueue, fb *storage.FileBuffer, interval time.Duration) *Watcher {
	return &Watcher{
		queue:    q,
		buffer:   fb,
		interval: interval,
		quit:     make(chan struct{}),
		done:     make(chan struct{}),
	}
}

// Start launches the sweep loop in its own goroutine. Call exactly once.
func (w *Watcher) Start() {
	go w.loop()
	log.Printf("watcher: sweeping for completed analytics every %s", w.interval)
}

// Stop terminates the sweep loop and blocks until it has exited.
func (w *Watcher) Stop() {
	close(w.quit)
	<-w.done
	log.Println("watcher: stopped")
}

func (w *Watcher) loop() {
	defer close(w.done)
	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()
	for {
		select {
		case <-w.quit:
			return
		case <-ticker.C:
			w.sweep()
		}
	}
}

// sweep completes every dispatched job whose analytics report exists.
func (w *Watcher) sweep() {
	for _, job := range w.queue.DispatchedJobs() {
		if w.buffer.AnalyticsReady(job.VideoID) {
			w.queue.MarkCompleted(job.ID)
			log.Printf("watcher: job %s completed (analytics ready for video %s)", job.ID, job.VideoID)
		}
	}
}
