# --- Build stage -----------------------------------------------------------
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /gateway ./cmd/server

# --- Runtime stage ----------------------------------------------------------
FROM alpine:3.20
RUN adduser -D -u 10001 gateway
COPY --from=build /gateway /usr/local/bin/gateway
USER gateway
ENV GATEWAY_PORT=8080 \
    DATA_DIR=/data
EXPOSE 8080
ENTRYPOINT ["gateway"]
