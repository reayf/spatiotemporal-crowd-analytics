package handlers

import (
	"errors"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	"spatiotemporal/gateway/internal/storage"
)

// AnalyticsHandler serves the mining engine's output back to API clients.
type AnalyticsHandler struct {
	buffer *storage.FileBuffer
}

// NewAnalyticsHandler builds the handler over the shared file buffer.
func NewAnalyticsHandler(fb *storage.FileBuffer) *AnalyticsHandler {
	return &AnalyticsHandler{buffer: fb}
}

// GetAnalytics handles GET /api/v1/analytics/:video_id.
//
// Returns the analytics report produced by the mining engine, or
// 202 Accepted while the pipeline has not published results yet — the
// client is expected to poll.
func (h *AnalyticsHandler) GetAnalytics(c *gin.Context) {
	videoID := c.Param("video_id")
	if !isSafeID(videoID) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid video id"})
		return
	}

	payload, err := h.buffer.ReadAnalytics(videoID)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			c.JSON(http.StatusAccepted, gin.H{
				"video_id": videoID,
				"status":   "PENDING",
				"detail":   "analytics not published yet; poll again later",
			})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to read analytics"})
		return
	}

	c.Data(http.StatusOK, "application/json", payload)
}
