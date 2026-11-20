// Package config centralises environment-driven configuration for the gateway.
package config

import (
	"os"
	"strconv"
)

// Config holds all runtime settings for the API gateway.
type Config struct {
	// Port is the HTTP listen port (without colon).
	Port string
	// DataDir is the root of the shared JSON file buffer that simulates the
	// message broker between the gateway and the Python engines.
	DataDir string
	// WorkerCount is the number of concurrent ingestion worker goroutines.
	WorkerCount int
	// MaxUploadMB caps accepted video upload size, in megabytes.
	MaxUploadMB int64
	// WatchIntervalSeconds is how often the completion watcher checks the
	// analytics directory for finished jobs.
	WatchIntervalSeconds int
}

// Load reads configuration from environment variables, falling back to
// sensible development defaults. The DataDir default assumes the server is
// started from the gateway-go/ directory (see README).
func Load() Config {
	return Config{
		Port:        getEnv("GATEWAY_PORT", "8080"),
		DataDir:     getEnv("DATA_DIR", "../data"),
		WorkerCount:          getEnvInt("WORKER_COUNT", 4),
		MaxUploadMB:          int64(getEnvInt("MAX_UPLOAD_MB", 512)),
		WatchIntervalSeconds: getEnvInt("WATCH_INTERVAL_SECONDS", 2),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return fallback
}
