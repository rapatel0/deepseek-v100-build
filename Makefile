SHELL := /usr/bin/env bash
.ONESHELL:

ifneq (,$(wildcard ./.env))
include .env
export
endif

IMAGE ?= local/deepseek-v4-flash:cuda-sm70
REGISTRY_IMAGE ?= localhost:32000/deepseek-v4-flash:cuda-sm70
PORT  ?= 8080

.PHONY: help download build push run stop logs k8s-apply k8s-delete k8s-restart smoke clean

help:
	@echo "Targets:"
	@echo "  download    Pull the 86.7 GB GGUF into MODEL_DIR (huggingface-cli)"
	@echo "  build       Build the CUDA image (run on gpu-01 or any linux/amd64+cuda host)"
	@echo "  push        Tag + push image to the in-cluster registry"
	@echo "  run         docker compose up -d"
	@echo "  stop        docker compose down"
	@echo "  logs        Tail llama-server logs"
	@echo "  k8s-apply   kubectl apply -k k8s/"
	@echo "  k8s-delete  kubectl delete -k k8s/"
	@echo "  smoke       Hit /v1/chat/completions (set K8S=1 for cluster)"
	@echo "  clean       Remove built image"

download:
	@bash scripts/download-model.sh

build:
	@bash scripts/build-image.sh

push:
	docker tag $(IMAGE) $(REGISTRY_IMAGE)
	docker push $(REGISTRY_IMAGE)

k8s-restart:
	kubectl -n deepseek rollout restart deploy/deepseek-v4-flash
	kubectl -n deepseek rollout status deploy/deepseek-v4-flash --timeout=10m

run:
	docker compose -f compose/docker-compose.yml --env-file .env up -d

stop:
	docker compose -f compose/docker-compose.yml --env-file .env down

logs:
	docker compose -f compose/docker-compose.yml --env-file .env logs -f --tail=200

k8s-apply:
	kubectl apply -k k8s/

k8s-delete:
	kubectl delete -k k8s/

smoke:
	@bash scripts/smoke-test.sh

clean:
	-docker image rm $(IMAGE)
