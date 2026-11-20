// Package router assembles the Gin engine and binds handlers to routes.
package router

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"spatiotemporal/gateway/internal/config"
	"spatiotemporal/gateway/internal/handlers"
	"spatiotemporal/gateway/internal/queue"
	"spatiotemporal/gateway/internal/storage"
)

// New builds the fully-wired HTTP router.
//
// Routes:
//
//	GET  /healthz                      liveness probe
//	POST /api/v1/videos                async video ingestion (multipart "video")
//	GET  /api/v1/jobs/:id              ingestion job status
//	GET  /api/v1/analytics/:video_id   mined spatiotemporal analytics
func New(cfg config.Config, q queue.Queue, fb *storage.FileBuffer) *gin.Engine {
	r := gin.New()
	r.Use(gin.Logger(), gin.Recovery())

	r.GET("/healthz", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	videoHandler := handlers.NewVideoHandler(q, fb, cfg.MaxUploadMB*1024*1024)
	analyticsHandler := handlers.NewAnalyticsHandler(fb)

	v1 := r.Group("/api/v1")
	{
		v1.POST("/videos", videoHandler.Ingest)
		v1.GET("/jobs/:id", videoHandler.GetJob)
		v1.GET("/analytics/:video_id", analyticsHandler.GetAnalytics)
	}

	return r
}
