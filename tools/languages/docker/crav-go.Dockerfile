# Go analysis image: the official toolchain (build/vet/test) + golangci-lint.
# Module + build caches are mounted at run time (/go/pkg/mod, /root/.cache/go-build).
FROM golang:1.22-alpine
RUN go install github.com/golangci/golangci-lint/cmd/golangci-lint@v1.59.1
