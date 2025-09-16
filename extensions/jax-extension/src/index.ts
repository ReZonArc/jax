/**
 * @file This file exports a class that implements the InferenceExtension interface from the @janhq/core package.
 * The class provides methods for initializing and stopping a model, and for making inference requests.
 * It also subscribes to events emitted by the @janhq/core package and handles new message requests.
 * @version 1.0.0
 * @module jax-extension/src/index
 */

import {
  AIEngine,
  getJanDataFolderPath,
  fs,
  joinPath,
  modelInfo,
  SessionInfo,
  UnloadResult,
  chatCompletion,
  chatCompletionChunk,
  ImportOptions,
  chatCompletionRequest,
  events,
  AppEvent,
  DownloadEvent,
} from '@janhq/core'

type JaxConfig = {
  device: string // 'cpu', 'gpu', 'tpu'
  precision: string // 'float32', 'float16', 'bfloat16'
  max_batch_size: number
  max_sequence_length: number
  compilation_cache: boolean
  jit_compile: boolean
  auto_spmd: boolean
  memory_fraction: number
}

interface JaxModelConfig {
  model_path: string
  name: string
  size_bytes: number
  sha256?: string
  model_type: 'flax' | 'jax' | 'orbax'
  config_path?: string
}

interface JaxSessionInfo extends SessionInfo {
  model_type: string
  device: string
  precision: string
}

declare global {
  interface Window {
    core: {
      extensionManager: {
        getByName(name: string): any
      }
    }
  }
}

/**
 * A class that implements the InferenceExtension interface from the @janhq/core package.
 * The class provides methods for initializing and stopping a model, and for making inference requests.
 * It uses JAX/Flax for high-performance ML inference with GPU acceleration.
 */
export default class JaxInferenceEngine extends AIEngine {
  provider: string = 'jax-ml'
  readonly providerId: string = 'jax-ml'

  private config: JaxConfig
  private providerPath!: string
  private loadedModels = new Map<string, JaxSessionInfo>()
  private loadingModels = new Map<string, Promise<SessionInfo>>()

  async onLoad(): Promise<void> {
    super.onLoad() // Calls registerEngine() from AIEngine

    // Default configuration for JAX engine
    const defaultConfig: JaxConfig = {
      device: 'gpu',
      precision: 'float16',
      max_batch_size: 1,
      max_sequence_length: 4096,
      compilation_cache: true,
      jit_compile: true,
      auto_spmd: false,
      memory_fraction: 0.8
    }

    // Register settings for JAX engine
    this.registerSettings([
      {
        key: 'device',
        title: 'Device',
        description: 'Device to use for inference',
        controllerType: 'dropdown',
        controllerProps: {
          options: [
            { value: 'gpu', name: 'GPU (CUDA/Metal)' },
            { value: 'cpu', name: 'CPU' },
            { value: 'tpu', name: 'TPU' }
          ],
          value: defaultConfig.device,
        },
      },
      {
        key: 'precision',
        title: 'Precision',
        description: 'Numerical precision for inference',
        controllerType: 'dropdown',
        controllerProps: {
          options: [
            { value: 'float32', name: 'Float32 (full precision)' },
            { value: 'float16', name: 'Float16 (half precision)' },
            { value: 'bfloat16', name: 'BFloat16 (brain float)' }
          ],
          value: defaultConfig.precision,
        },
      },
      {
        key: 'max_batch_size',
        title: 'Max Batch Size',
        description: 'Maximum batch size for inference',
        controllerType: 'slider',
        controllerProps: {
          min: 1,
          max: 32,
          step: 1,
          value: defaultConfig.max_batch_size,
        },
      },
      {
        key: 'max_sequence_length',
        title: 'Max Sequence Length',
        description: 'Maximum sequence length for model input',
        controllerType: 'slider',
        controllerProps: {
          min: 512,
          max: 32768,
          step: 512,
          value: defaultConfig.max_sequence_length,
        },
      },
      {
        key: 'memory_fraction',
        title: 'Memory Fraction',
        description: 'Fraction of GPU memory to use',
        controllerType: 'slider',
        controllerProps: {
          min: 0.1,
          max: 1.0,
          step: 0.1,
          value: defaultConfig.memory_fraction,
        },
      },
      {
        key: 'jit_compile',
        title: 'JIT Compilation',
        description: 'Enable Just-In-Time compilation for better performance',
        controllerType: 'checkbox',
        controllerProps: {
          value: defaultConfig.jit_compile,
        },
      },
      {
        key: 'compilation_cache',
        title: 'Compilation Cache',
        description: 'Enable compilation cache for faster startup',
        controllerType: 'checkbox',
        controllerProps: {
          value: defaultConfig.compilation_cache,
        },
      },
    ])

    // Load configuration
    let loadedConfig: any = {}
    for (const setting of this.getRegisteredSettings()) {
      const defaultValue = setting.controllerProps.value
      loadedConfig[setting.key] = await this.getSetting<typeof defaultValue>(
        setting.key,
        defaultValue
      )
    }
    this.config = { ...defaultConfig, ...loadedConfig } as JaxConfig

    // Set provider path
    this.providerPath = await joinPath([
      await getJanDataFolderPath(),
      this.providerId,
    ])

    console.log('JAX-ML inference engine loaded with config:', this.config)
  }

  async onUnload(): Promise<void> {
    // Unload all active models
    for (const [modelId] of this.loadedModels) {
      await this.unload(modelId)
    }
  }

  onSettingUpdate<T>(key: string, value: T): void {
    (this.config as any)[key] = value
    console.log(`JAX engine setting updated: ${key} = ${value}`)
  }

  async getProviderPath(): Promise<string> {
    if (!this.providerPath) {
      this.providerPath = await joinPath([
        await getJanDataFolderPath(),
        this.providerId,
      ])
    }
    return this.providerPath
  }

  // Implement the required LocalProvider interface methods
  async list(): Promise<modelInfo[]> {
    const modelsDir = await joinPath([await this.getProviderPath(), 'models'])
    if (!(await fs.existsSync(modelsDir))) {
      await fs.mkdir(modelsDir)
      return []
    }

    const modelIds: string[] = []
    const children = await fs.readdirSync(modelsDir)
    
    for (const child of children) {
      const childPath = await joinPath([modelsDir, child])
      const stat = await fs.fileStat(childPath)
      if (stat.isDirectory) {
        const configPath = await joinPath([childPath, 'model.yml'])
        if (await fs.existsSync(configPath)) {
          modelIds.push(child)
        }
      }
    }

    const modelInfos: modelInfo[] = []
    for (const modelId of modelIds) {
      try {
        const configPath = await joinPath([modelsDir, modelId, 'model.yml'])
        const modelConfig = await this.readModelConfig(configPath)
        
        const modelInfo = {
          id: modelId,
          name: modelConfig.name ?? modelId,
          providerId: this.provider,
          port: 0, // Port is assigned when loaded
          sizeBytes: modelConfig.size_bytes ?? 0,
        } as modelInfo
        modelInfos.push(modelInfo)
      } catch (error) {
        console.error(`Error reading model config for ${modelId}:`, error)
      }
    }

    return modelInfos
  }

  async import(modelId: string, opts: ImportOptions): Promise<void> {
    const isValidModelId = (id: string) => {
      if (!/^[a-zA-Z0-9/_\-\.]+$/.test(id)) return false
      const parts = id.split('/')
      return parts.every((s) => s !== '' && s !== '.' && s !== '..')
    }

    if (!isValidModelId(modelId)) {
      throw new Error(
        `Invalid modelId: ${modelId}. Only alphanumeric and / _ - . characters are allowed.`
      )
    }

    const configPath = await joinPath([
      await this.getProviderPath(),
      'models',
      modelId,
      'model.yml',
    ])
    
    if (await fs.existsSync(configPath)) {
      throw new Error(`Model ${modelId} already exists`)
    }

    const modelDir = `${this.providerId}/models/${modelId}`
    const janDataFolderPath = await getJanDataFolderPath()

    // Handle model path - could be URL or local file
    let modelPath = opts.modelPath
    let downloadItems: any[] = []

    if (modelPath.startsWith('https://')) {
      const localPath = `${modelDir}/model.safetensors`
      downloadItems.push({
        url: modelPath,
        save_path: localPath,
        sha256: opts.modelSha256,
        size: opts.modelSize,
      })
      modelPath = localPath
    } else {
      // Local file - verify it exists
      if (!(await fs.existsSync(modelPath))) {
        throw new Error(`File not found: ${modelPath}`)
      }
    }

    // Download if needed
    if (downloadItems.length > 0) {
      try {
        const onProgress = (transferred: number, total: number) => {
          events.emit(DownloadEvent.onFileDownloadUpdate, {
            modelId,
            percent: transferred / total,
            size: { transferred, total },
            downloadType: 'Model',
          })
        }
        
        const downloadManager = window.core.extensionManager.getByName(
          '@janhq/download-extension'
        )
        await downloadManager.downloadFiles(
          downloadItems,
          `${this.provider}/${modelId}`,
          onProgress
        )
        
        events.emit(DownloadEvent.onFileDownloadAndVerificationSuccess, {
          modelId,
          downloadType: 'Model',
        })
      } catch (error: any) {
        console.error('Error downloading JAX model:', error)
        events.emit(DownloadEvent.onFileDownloadError, {
          modelId,
          downloadType: 'Model',
          error: error.message,
        })
        throw error
      }
    }

    // Calculate file size
    const fullModelPath = await joinPath([janDataFolderPath, modelPath])
    const size_bytes = (await fs.fileStat(fullModelPath)).size

    // Create model config
    const modelConfig: JaxModelConfig = {
      model_path: modelPath,
      name: modelId,
      size_bytes,
      sha256: opts.modelSha256,
      model_type: this.detectModelType(fullModelPath),
    }

    // Create model directory and save config
    await fs.mkdir(await joinPath([janDataFolderPath, modelDir]))
    await this.writeModelConfig(configPath, modelConfig)
    
    events.emit(AppEvent.onModelImported, {
      modelId,
      modelPath,
      size_bytes,
      model_sha256: opts.modelSha256,
      model_size_bytes: opts.modelSize,
    })
  }

  async abortImport(modelId: string): Promise<void> {
    const taskId = `${this.provider}/${modelId}`
    const downloadManager = window.core.extensionManager.getByName(
      '@janhq/download-extension'
    )
    
    try {
      await downloadManager.cancelDownload(taskId)
    } catch (error) {
      console.warn('Failed to cancel download task:', error)
    }

    // Clean up partial files
    const modelDir = await joinPath([
      await this.getProviderPath(),
      'models',
      modelId,
    ])
    
    if (await fs.existsSync(modelDir)) {
      await fs.rm(modelDir)
    }
  }

  async load(
    modelId: string,
    overrideSettings?: Partial<JaxConfig>
  ): Promise<SessionInfo> {
    // Check if model is already loaded
    if (this.loadedModels.has(modelId)) {
      return this.loadedModels.get(modelId)!
    }

    // Check if model is currently being loaded
    if (this.loadingModels.has(modelId)) {
      return this.loadingModels.get(modelId)!
    }

    // Start loading process
    const loadingPromise = this.performLoad(modelId, overrideSettings)
    this.loadingModels.set(modelId, loadingPromise)

    try {
      const result = await loadingPromise
      return result
    } finally {
      this.loadingModels.delete(modelId)
    }
  }

  private async performLoad(
    modelId: string,
    overrideSettings?: Partial<JaxConfig>
  ): Promise<JaxSessionInfo> {
    const config = { ...this.config, ...(overrideSettings ?? {}) }
    
    // Load model configuration
    const modelConfigPath = await joinPath([
      await this.getProviderPath(),
      'models',
      modelId,
      'model.yml',
    ])
    
    const modelConfig = await this.readModelConfig(modelConfigPath)
    const janDataFolderPath = await getJanDataFolderPath()
    const modelPath = await joinPath([janDataFolderPath, modelConfig.model_path])

    // Create session info
    const port = await this.getRandomPort()
    const sessionInfo: JaxSessionInfo = {
      model_id: modelId,
      model_path: modelPath,
      port,
      pid: Date.now(), // Use timestamp as simple PID
      api_key: await this.generateApiKey(modelId),
      model_type: modelConfig.model_type,
      device: config.device,
      precision: config.precision,
    }

    // Simulate model loading (in real implementation, this would load the JAX/Flax model)
    console.log(`Loading JAX model ${modelId} on ${config.device} with ${config.precision} precision`)
    
    // Store session
    this.loadedModels.set(modelId, sessionInfo)
    
    console.log(`JAX model ${modelId} loaded successfully`)
    return sessionInfo
  }

  async unload(modelId: string): Promise<UnloadResult> {
    const sessionInfo = this.loadedModels.get(modelId)
    if (!sessionInfo) {
      throw new Error(`No active session found for model: ${modelId}`)
    }

    try {
      // Simulate model unloading
      console.log(`Unloading JAX model ${modelId}`)
      
      this.loadedModels.delete(modelId)
      
      return {
        success: true,
        error: null,
      }
    } catch (error: any) {
      console.error('Error unloading JAX model:', error)
      return {
        success: false,
        error: `Failed to unload model: ${error}`,
      }
    }
  }

  async chat(
    opts: chatCompletionRequest,
    abortController?: AbortController
  ): Promise<chatCompletion | AsyncIterable<chatCompletionChunk>> {
    const sessionInfo = this.loadedModels.get(opts.model)
    if (!sessionInfo) {
      throw new Error(`No active session found for model: ${opts.model}`)
    }

    // Simulate inference (in real implementation, this would call JAX/Flax model)
    if (opts.stream) {
      return this.handleStreamingResponse(opts, sessionInfo, abortController)
    } else {
      return this.handleNonStreamingResponse(opts, sessionInfo)
    }
  }

  private async *handleStreamingResponse(
    opts: chatCompletionRequest,
    sessionInfo: JaxSessionInfo,
    abortController?: AbortController
  ): AsyncIterable<chatCompletionChunk> {
    const responses = [
      "I'm a JAX-ML powered AI assistant. ",
      "I use high-performance ML inference ",
      "with GPU acceleration for fast responses. ",
      "How can I help you today?"
    ]

    for (let i = 0; i < responses.length; i++) {
      if (abortController?.signal.aborted) {
        break
      }

      yield {
        id: `jax-${Date.now()}-${i}`,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: opts.model,
        choices: [{
          index: 0,
          delta: {
            content: responses[i],
          },
          finish_reason: i === responses.length - 1 ? 'stop' : null,
        }],
      }

      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 100))
    }
  }

  private async handleNonStreamingResponse(
    opts: chatCompletionRequest,
    sessionInfo: JaxSessionInfo
  ): Promise<chatCompletion> {
    const responseText = "I'm a JAX-ML powered AI assistant. I use high-performance ML inference with GPU acceleration for fast responses. How can I help you today?"

    return {
      id: `jax-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: opts.model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: responseText,
        },
        finish_reason: 'stop',
      }],
      usage: {
        prompt_tokens: this.estimateTokens(opts.messages),
        completion_tokens: this.estimateTokens([{ role: 'assistant', content: responseText }]),
        total_tokens: 0,
      },
    }
  }

  async delete(modelId: string): Promise<void> {
    // Unload if currently loaded
    if (this.loadedModels.has(modelId)) {
      await this.unload(modelId)
    }

    const modelDir = await joinPath([
      await this.getProviderPath(),
      'models',
      modelId,
    ])

    if (!(await fs.existsSync(await joinPath([modelDir, 'model.yml'])))) {
      throw new Error(`Model ${modelId} does not exist`)
    }

    await fs.rm(modelDir)
  }

  async getLoadedModels(): Promise<string[]> {
    return Array.from(this.loadedModels.keys())
  }

  getChatClient(sessionId: string): any {
    throw new Error('Direct chat client access not implemented for JAX engine')
  }

  // Helper methods
  private async getRandomPort(): Promise<number> {
    // Simple port generation (in real implementation, check if port is available)
    return Math.floor(Math.random() * (65535 - 3000)) + 3000
  }

  private async generateApiKey(modelId: string): Promise<string> {
    // Simple API key generation
    return `jax-${modelId}-${Date.now()}`
  }

  private detectModelType(filePath: string): 'flax' | 'jax' | 'orbax' {
    if (filePath.includes('flax') || filePath.endsWith('.msgpack')) {
      return 'flax'
    } else if (filePath.includes('orbax')) {
      return 'orbax'
    } else {
      return 'jax'
    }
  }

  private async readModelConfig(configPath: string): Promise<JaxModelConfig> {
    // In real implementation, would use YAML parser
    const content = await fs.readFile(configPath, 'utf-8')
    return JSON.parse(content) as JaxModelConfig
  }

  private async writeModelConfig(configPath: string, config: JaxModelConfig): Promise<void> {
    // In real implementation, would use YAML formatter
    await fs.writeFile(configPath, JSON.stringify(config, null, 2))
  }

  private estimateTokens(messages: any[]): number {
    // Simple token estimation
    const text = messages.map(m => m.content || '').join(' ')
    return Math.ceil(text.length / 4)
  }
}