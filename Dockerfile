# syntax=docker/dockerfile:1

# ---------- Stage 1: build ----------
FROM golang:1.26-alpine AS build

ARG VERSION=dev
ARG COMMIT=unknown
ARG BUILD_DATE=unknown

WORKDIR /src

# Cache deps first
COPY go/go.mod go/go.sum ./go/
RUN cd go && go mod download

COPY go/ ./go/

RUN cd go && CGO_ENABLED=0 go build \
      -ldflags="-s -w \
        -X main.version=${VERSION} \
        -X main.commit=${COMMIT} \
        -X main.date=${BUILD_DATE}" \
      -o /usr/local/bin/fit \
      ./cmd/fit

# ---------- Stage 2: runtime ----------
FROM gcr.io/distroless/static-debian12

LABEL org.opencontainers.image.title="fit" \
      org.opencontainers.image.description="Train small advisor models to steer black-box LLMs" \
      org.opencontainers.image.url="https://github.com/hop-top/fit" \
      org.opencontainers.image.source="https://github.com/hop-top/fit" \
      org.opencontainers.image.vendor="hop-top" \
      org.opencontainers.image.licenses="MIT"

COPY --from=build /usr/local/bin/fit /usr/local/bin/fit

ENTRYPOINT ["/usr/local/bin/fit"]
