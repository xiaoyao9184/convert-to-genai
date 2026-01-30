
group "default" {
  targets = ["cpu", "cuda"]
}

target "cpu" {
  args = {
    GENAI_VERSION = "0.11.2"
  }
  cache-from = [
    {
      type = "local"
      src = ".buildx-cache"
    }
  ]
  cache-to = [
    {
      type = "local"
      dest = ".buildx-cache"
      mode = "max"
    }
  ]
  context = "."
  dockerfile = "docker/build.python@cpu/dockerfile"
  output = [
    {
      type = "image"
    }
  ]
  platforms = ["linux/amd64", "linux/arm64"]
  tags = ["xiaoyao9184/convert-to-genai:cpu"]
}

target "cuda" {
  args = {
    GENAI_VERSION = "0.11.2"
  }
  cache-from = [
    {
      type = "local"
      src = ".buildx-cache"
    }
  ]
  cache-to = [
    {
      type = "local"
      dest = ".buildx-cache"
      mode = "max"
    }
  ]
  context = "."
  dockerfile = "docker/build.pytorch@cuda/dockerfile"
  output = [
    {
      type = "image"
    }
  ]
  platforms = ["linux/amd64"]
  tags = ["xiaoyao9184/convert-to-genai:cuda"]
}
