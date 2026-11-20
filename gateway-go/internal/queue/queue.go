// Package queue provides the asynchronous ingestion pipeline: an in-memory,
// channel-backed job queue drained by a pool of worker goroutines.
//
// In production this would be RabbitMQ/Kafka; here the channel gives us the
// async boundary inside the gateway, and the JSON file buffer gives us the
// cross-service boundary towards the Python engines.
package queue

import (
	"errors"
	"log"
	"sync"
	"time"

	"spatiotemporal/gateway/internal/domain"
	"spatiotemporal/gateway/internal/storage"
)

// ErrQueueFull is returned when the pending buffer is saturated; callers
// should translate it to HTTP 503 so clients can retry with backoff.
var ErrQueueFull = errors.New("queue: pending buffer is full")

// pendingBufferSize bounds how many accepted-but-undispatched jobs may pile
// up before the gateway starts shedding load.
const pendingBufferSize = 128

// Queue is the contract the HTTP layer depends on (Dependency Inversion:
// handlers never see the concrete worker pool).
type Queue interface {
	// Enqueue registers a job and schedules it for asynchronous dispatch.
	Enqueue(job *domain.Job) error
	// Get returns a snapshot of a job's current state.
	Get(id string) (domain.Job, bool)
}

// InMemoryQueue implements Queue with a mutex-guarded job registry and a
// buffered channel drained by WorkerCount goroutines.
type InMemoryQueue struct {
	buffer  *storage.FileBuffer
	workers int

	mu      sync.RWMutex
	jobs    map[string]*domain.Job
	pending chan string
	quit    chan struct{}
	wg      sync.WaitGroup
}

// NewInMemoryQueue wires the queue to the file buffer it dispatches into.
func NewInMemoryQueue(workers int, buffer *storage.FileBuffer) *InMemoryQueue {
	return &InMemoryQueue{
		buffer:  buffer,
		workers: workers,
		jobs:    make(map[string]*domain.Job),
		pending: make(chan string, pendingBufferSize),
		quit:    make(chan struct{}),
	}
}

// Start launches the worker pool. Call exactly once.
func (q *InMemoryQueue) Start() {
	for i := 1; i <= q.workers; i++ {
		q.wg.Add(1)
		go q.worker(i)
	}
	log.Printf("queue: started %d ingestion workers", q.workers)
}

// Stop signals all workers to exit and blocks until they have drained.
func (q *InMemoryQueue) Stop() {
	close(q.quit)
	q.wg.Wait()
	log.Println("queue: all workers stopped")
}

// Enqueue implements Queue. It never blocks the HTTP request path: if the
// pending buffer is full the job is rejected immediately.
func (q *InMemoryQueue) Enqueue(job *domain.Job) error {
	q.mu.Lock()
	q.jobs[job.ID] = job
	q.mu.Unlock()

	select {
	case q.pending <- job.ID:
		return nil
	default:
		q.setStatus(job.ID, domain.StatusFailed, ErrQueueFull.Error())
		return ErrQueueFull
	}
}

// Get implements Queue, returning a copy so callers cannot mutate shared state.
func (q *InMemoryQueue) Get(id string) (domain.Job, bool) {
	q.mu.RLock()
	defer q.mu.RUnlock()
	job, ok := q.jobs[id]
	if !ok {
		return domain.Job{}, false
	}
	return *job, true
}

// worker drains the pending channel until Stop is called.
func (q *InMemoryQueue) worker(id int) {
	defer q.wg.Done()
	for {
		select {
		case <-q.quit:
			return
		case jobID := <-q.pending:
			q.dispatch(id, jobID)
		}
	}
}

// dispatch hands a job off to the CV engine through the file buffer.
func (q *InMemoryQueue) dispatch(workerID int, jobID string) {
	job, ok := q.Get(jobID)
	if !ok {
		return
	}
	log.Printf("queue: [worker %d] dispatching job %s (video %s)", workerID, job.ID, job.VideoID)

	if err := q.buffer.WriteTrackingManifest(&job); err != nil {
		log.Printf("queue: [worker %d] job %s failed: %v", workerID, job.ID, err)
		q.setStatus(jobID, domain.StatusFailed, err.Error())
		return
	}
	q.setStatus(jobID, domain.StatusDispatched, "")
	// From here the pipeline is owned by the Python engines; the completion
	// watcher (internal/watcher) flips the job to COMPLETED once the mining
	// engine publishes data/analytics/<video_id>.json.
}

// DispatchedJobs returns snapshots of all jobs waiting for analytics,
// for the completion watcher to sweep.
func (q *InMemoryQueue) DispatchedJobs() []domain.Job {
	q.mu.RLock()
	defer q.mu.RUnlock()
	jobs := make([]domain.Job, 0)
	for _, job := range q.jobs {
		if job.Status == domain.StatusDispatched {
			jobs = append(jobs, *job)
		}
	}
	return jobs
}

// MarkCompleted flips a dispatched job to its terminal COMPLETED state.
func (q *InMemoryQueue) MarkCompleted(jobID string) {
	q.setStatus(jobID, domain.StatusCompleted, "")
}

// setStatus atomically updates a job's status, error message and timestamp.
func (q *InMemoryQueue) setStatus(jobID string, status domain.JobStatus, errMsg string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if job, ok := q.jobs[jobID]; ok {
		job.Status = status
		job.Error = errMsg
		job.UpdatedAt = time.Now().UTC()
	}
}
