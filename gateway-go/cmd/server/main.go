// Command server runs the Spatiotemporal Tracking API gateway.
//
// The gateway accepts video uploads, dispatches asynchronous tracking jobs
// to the Python CV engine through a JSON file buffer (simulated message
// queue), and serves the mining engine's analytics over REST.
package main

import (
	"context"
	"errors"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"spatiotemporal/gateway/internal/config"
	"spatiotemporal/gateway/internal/queue"
	"spatiotemporal/gateway/internal/router"
	"spatiotemporal/gateway/internal/storage"
	"spatiotemporal/gateway/internal/watcher"
)

func main() {
	cfg := config.Load()

	buffer, err := storage.NewFileBuffer(cfg.DataDir)
	if err != nil {
		log.Fatalf("gateway: init file buffer: %v", err)
	}

	jobQueue := queue.NewInMemoryQueue(cfg.WorkerCount, buffer)
	jobQueue.Start()

	completionWatcher := watcher.New(
		jobQueue, buffer, time.Duration(cfg.WatchIntervalSeconds)*time.Second,
	)
	completionWatcher.Start()

	srv := &http.Server{
		Addr:              ":" + cfg.Port,
		Handler:           router.New(cfg, jobQueue, buffer),
		ReadHeaderTimeout: 10 * time.Second,
	}

	go func() {
		log.Printf("gateway: listening on :%s (data dir: %s)", cfg.Port, cfg.DataDir)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("gateway: server error: %v", err)
		}
	}()

	// Block until SIGINT/SIGTERM, then drain workers and in-flight requests.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("gateway: shutdown signal received")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	completionWatcher.Stop()
	jobQueue.Stop()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("gateway: forced shutdown: %v", err)
	}
	log.Println("gateway: stopped cleanly")
}
