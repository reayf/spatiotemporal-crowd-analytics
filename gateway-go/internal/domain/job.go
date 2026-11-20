// Package domain defines the core entities shared across the gateway layers.
package domain

import "time"

// JobStatus enumerates the lifecycle states of a video ingestion job.
type JobStatus string

const (
	// StatusQueued means the job was accepted and is waiting for a worker.
	StatusQueued JobStatus = "QUEUED"
	// StatusDispatched means the job manifest was written to the file buffer
	// and is now owned by the CV engine.
	StatusDispatched JobStatus = "DISPATCHED"
	// StatusCompleted means analytics for the video have been observed on disk.
	StatusCompleted JobStatus = "COMPLETED"
	// StatusFailed is a terminal error during ingestion or dispatch.
	StatusFailed JobStatus = "FAILED"
)

// Job represents one video ingestion request travelling through the pipeline:
// upload -> tracking (CV engine) -> mining (analytics engine).
type Job struct {
	ID        string    `json:"job_id"`
	VideoID   string    `json:"video_id"`
	VideoPath string    `json:"video_path"`
	Status    JobStatus `json:"status"`
	Error     string    `json:"error,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}
