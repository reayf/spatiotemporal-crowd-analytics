// Package handlers contains the Gin HTTP handlers for the gateway.
package handlers

import (
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"spatiotemporal/gateway/internal/domain"
	"spatiotemporal/gateway/internal/ids"
	"spatiotemporal/gateway/internal/queue"
	"spatiotemporal/gateway/internal/storage"
)

// VideoHandler exposes video ingestion and job status endpoints.
type VideoHandler struct {
	queue          queue.Queue
	buffer         *storage.FileBuffer
	maxUploadBytes int64
}

// NewVideoHandler wires the handler to its dependencies via interfaces where
// behaviour matters (queue) and a concrete adapter where it is pure IO.
func NewVideoHandler(q queue.Queue, fb *storage.FileBuffer, maxUploadBytes int64) *VideoHandler {
	return &VideoHandler{queue: q, buffer: fb, maxUploadBytes: maxUploadBytes}
}

// Ingest handles POST /api/v1/videos.
//
// Accepts a multipart form with a "video" file field, persists the file,
// and enqueues an asynchronous tracking job. Responds 202 Accepted with the
// job and video identifiers the client should poll with.
func (h *VideoHandler) Ingest(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, h.maxUploadBytes)

	fileHeader, err := c.FormFile("video")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "multipart field 'video' is required and must fit the upload limit",
		})
		return
	}

	src, err := fileHeader.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read upload"})
		return
	}
	defer src.Close()

	videoID := ids.New()
	videoPath, err := h.buffer.SaveVideo(videoID, src, fileHeader.Filename)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to persist video"})
		return
	}

	now := time.Now().UTC()
	job := &domain.Job{
		ID:        ids.New(),
		VideoID:   videoID,
		VideoPath: videoPath,
		Status:    domain.StatusQueued,
		CreatedAt: now,
		UpdatedAt: now,
	}
	if err := h.queue.Enqueue(job); err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "ingestion queue is full, retry later"})
		return
	}

	c.JSON(http.StatusAccepted, gin.H{
		"job_id":   job.ID,
		"video_id": job.VideoID,
		"status":   job.Status,
	})
}

// GetJob handles GET /api/v1/jobs/:id and returns the job's current state.
func (h *VideoHandler) GetJob(c *gin.Context) {
	id := c.Param("id")
	if !isSafeID(id) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid job id"})
		return
	}
	job, ok := h.queue.Get(id)
	if !ok {
		c.JSON(http.StatusNotFound, gin.H{"error": "job not found"})
		return
	}
	c.JSON(http.StatusOK, job)
}

// isSafeID rejects identifiers that could escape the data directories when
// interpolated into file paths.
func isSafeID(id string) bool {
	return id != "" && !strings.ContainsAny(id, `/\.`)
}
