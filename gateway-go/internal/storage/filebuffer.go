// Package storage implements the JSON file buffer that simulates a message
// queue between the Go gateway and the Python engines.
//
// Layout under the configured data root:
//
//	data/
//	├── videos/          raw uploaded videos
//	├── queue/tracking/  job manifests consumed by the CV engine
//	├── queue/mining/    job manifests consumed by the mining engine
//	├── trajectories/    per-video trajectory logs (CV engine output)
//	└── analytics/       per-video analytics reports (mining engine output)
package storage

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"spatiotemporal/gateway/internal/domain"
)

// FileBuffer is the gateway's adapter over the shared exchange directories.
type FileBuffer struct {
	root string
}

// NewFileBuffer resolves the data root and ensures all exchange directories
// exist so the Python engines can start polling immediately.
func NewFileBuffer(root string) (*FileBuffer, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("storage: resolve data dir %q: %w", root, err)
	}
	fb := &FileBuffer{root: abs}
	for _, dir := range []string{
		fb.videosDir(),
		fb.trackingQueueDir(),
		filepath.Join(abs, "queue", "mining"),
		filepath.Join(abs, "trajectories"),
		fb.analyticsDir(),
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("storage: create %s: %w", dir, err)
		}
	}
	return fb, nil
}

func (fb *FileBuffer) videosDir() string        { return filepath.Join(fb.root, "videos") }
func (fb *FileBuffer) trackingQueueDir() string { return filepath.Join(fb.root, "queue", "tracking") }
func (fb *FileBuffer) analyticsDir() string     { return filepath.Join(fb.root, "analytics") }

// SaveVideo streams an uploaded video to disk under a stable, ID-based name
// and returns the absolute path the CV engine should read from.
func (fb *FileBuffer) SaveVideo(videoID string, src io.Reader, originalName string) (string, error) {
	ext := filepath.Ext(originalName)
	if ext == "" {
		ext = ".mp4"
	}
	dstPath := filepath.Join(fb.videosDir(), videoID+ext)

	dst, err := os.Create(dstPath)
	if err != nil {
		return "", fmt.Errorf("storage: create video file: %w", err)
	}
	defer dst.Close()

	if _, err := io.Copy(dst, src); err != nil {
		os.Remove(dstPath) // don't leave partial uploads behind
		return "", fmt.Errorf("storage: write video file: %w", err)
	}
	return dstPath, nil
}

// WriteTrackingManifest publishes a job to the CV engine's queue directory.
// This is the asynchronous hand-off that simulates producing to a broker.
// The write is atomic (tmp file + rename) so consumers never observe a
// half-written manifest.
func (fb *FileBuffer) WriteTrackingManifest(job *domain.Job) error {
	payload, err := json.MarshalIndent(job, "", "  ")
	if err != nil {
		return fmt.Errorf("storage: marshal manifest: %w", err)
	}

	finalPath := filepath.Join(fb.trackingQueueDir(), job.ID+".json")
	tmpPath := finalPath + ".tmp"
	if err := os.WriteFile(tmpPath, payload, 0o644); err != nil {
		return fmt.Errorf("storage: write manifest: %w", err)
	}
	if err := os.Rename(tmpPath, finalPath); err != nil {
		return fmt.Errorf("storage: commit manifest: %w", err)
	}
	return nil
}

// ReadAnalytics returns the raw analytics JSON produced by the mining engine
// for the given video, or an error wrapping os.ErrNotExist when the report
// is not (yet) available.
func (fb *FileBuffer) ReadAnalytics(videoID string) ([]byte, error) {
	return os.ReadFile(filepath.Join(fb.analyticsDir(), videoID+".json"))
}

// AnalyticsReady reports whether the mining engine has published a report
// for the given video.
func (fb *FileBuffer) AnalyticsReady(videoID string) bool {
	_, err := os.Stat(filepath.Join(fb.analyticsDir(), videoID+".json"))
	return err == nil
}
