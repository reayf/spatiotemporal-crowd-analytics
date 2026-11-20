// Package ids provides dependency-free random identifiers.
package ids

import (
	"crypto/rand"
	"encoding/hex"
)

// New returns a 32-character random hex identifier (128 bits of entropy,
// UUID-equivalent) without pulling in an external dependency.
func New() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand failing means the OS entropy source is broken;
		// there is no sane way to continue serving requests.
		panic("ids: crypto/rand unavailable: " + err.Error())
	}
	return hex.EncodeToString(b)
}
