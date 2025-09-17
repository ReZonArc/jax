import { describe, it, expect, beforeEach, vi } from 'vitest'
import JaxInferenceEngine from '../index'

// Mock the core dependencies
vi.mock('@janhq/core', () => ({
  AIEngine: class MockAIEngine {
    registerEngine() {}
    registerSettings() {}
    onLoad() {}
    getSetting = vi.fn().mockResolvedValue('default')
    getSettings = vi.fn().mockResolvedValue([])
    updateSettings = vi.fn()
    getRegisteredSettings = vi.fn().mockReturnValue([])
  },
  getJanDataFolderPath: vi.fn().mockResolvedValue('/mock/data'),
  joinPath: vi.fn().mockImplementation((...paths) => paths.join('/')),
  fs: {
    existsSync: vi.fn().mockResolvedValue(true),
    mkdir: vi.fn(),
    readdirSync: vi.fn().mockResolvedValue([]),
    fileStat: vi.fn().mockResolvedValue({ isDirectory: true, size: 1000 }),
    readFile: vi.fn().mockResolvedValue('{}'),
    writeFile: vi.fn(),
    rm: vi.fn(),
  },
  events: {
    emit: vi.fn(),
  },
  AppEvent: {
    onModelImported: 'onModelImported',
  },
  DownloadEvent: {
    onFileDownloadUpdate: 'onFileDownloadUpdate',
    onFileDownloadAndVerificationSuccess: 'onFileDownloadAndVerificationSuccess',
    onFileDownloadError: 'onFileDownloadError',
  },
}))

describe('JaxInferenceEngine', () => {
  let engine: JaxInferenceEngine

  beforeEach(() => {
    engine = new JaxInferenceEngine()
    // Initialize config to avoid undefined errors
    engine['config'] = {
      device: 'gpu',
      precision: 'float16',
      max_batch_size: 1,
      max_sequence_length: 4096,
      compilation_cache: true,
      jit_compile: true,
      auto_spmd: false,
      memory_fraction: 0.8
    }
    vi.clearAllMocks()
  })

  it('should initialize with correct provider name', () => {
    expect(engine.provider).toBe('jax-ml')
    expect(engine.providerId).toBe('jax-ml')
  })

  it('should load successfully', async () => {
    await expect(engine.onLoad()).resolves.not.toThrow()
  })

  it('should list models when directory exists', async () => {
    const models = await engine.list()
    expect(Array.isArray(models)).toBe(true)
  })

  it('should handle model import', async () => {
    // Mock that config file doesn't exist initially, but local file exists
    const mockFs = await import('@janhq/core')
    vi.mocked(mockFs.fs.existsSync)
      .mockResolvedValueOnce(false) // Config file doesn't exist
      .mockResolvedValueOnce(true)  // Local model file exists
    
    const opts = {
      modelPath: '/test/model.safetensors',
      modelSha256: 'test-hash',
      modelSize: 1000,
    }

    await expect(engine.import('test-model', opts)).resolves.not.toThrow()
  })

  it('should validate model IDs correctly', async () => {
    const invalidModelId = '../invalid'
    const opts = {
      modelPath: '/test/model.safetensors',
    }

    await expect(engine.import(invalidModelId, opts)).rejects.toThrow('Invalid modelId')
  })

  it('should load and unload models', async () => {
    const modelId = 'test-model'
    
    // Mock model config
    const mockFs = await import('@janhq/core')
    vi.mocked(mockFs.fs.readFile).mockResolvedValue(JSON.stringify({
      model_path: '/test/model.safetensors',
      name: 'Test Model',
      size_bytes: 1000,
      model_type: 'flax'
    }))

    const sessionInfo = await engine.load(modelId)
    expect(sessionInfo.model_id).toBe(modelId)
    expect(typeof sessionInfo.port).toBe('number')
    expect(typeof sessionInfo.api_key).toBe('string')

    const unloadResult = await engine.unload(modelId)
    expect(unloadResult.success).toBe(true)
  })

  it('should handle chat completion (non-streaming)', async () => {
    const modelId = 'test-model'
    
    // First load the model
    const mockFs = await import('@janhq/core')
    vi.mocked(mockFs.fs.readFile).mockResolvedValue(JSON.stringify({
      model_path: '/test/model.safetensors',
      name: 'Test Model',
      size_bytes: 1000,
      model_type: 'flax'
    }))

    await engine.load(modelId)

    const chatRequest = {
      model: modelId,
      messages: [{ role: 'user', content: 'Hello' }],
      stream: false,
    }

    const response = await engine.chat(chatRequest)
    expect(response).toHaveProperty('id')
    expect(response).toHaveProperty('object', 'chat.completion')
    expect(response).toHaveProperty('choices')
    expect(Array.isArray((response as any).choices)).toBe(true)
  })

  it('should handle chat completion (streaming)', async () => {
    const modelId = 'test-model'
    
    // First load the model
    const mockFs = await import('@janhq/core')
    vi.mocked(mockFs.fs.readFile).mockResolvedValue(JSON.stringify({
      model_path: '/test/model.safetensors',
      name: 'Test Model',
      size_bytes: 1000,
      model_type: 'flax'
    }))

    await engine.load(modelId)

    const chatRequest = {
      model: modelId,
      messages: [{ role: 'user', content: 'Hello' }],
      stream: true,
    }

    const response = await engine.chat(chatRequest)
    expect(typeof response[Symbol.asyncIterator]).toBe('function')

    // Test streaming chunks
    const chunks = []
    for await (const chunk of response as AsyncIterable<any>) {
      chunks.push(chunk)
      if (chunks.length > 10) break // Prevent infinite loop in test
    }

    expect(chunks.length).toBeGreaterThan(0)
    expect(chunks[0]).toHaveProperty('object', 'chat.completion.chunk')
  })

  it('should delete models', async () => {
    const modelId = 'test-model'
    // Mock that model.yml exists
    const mockFs = await import('@janhq/core')
    vi.mocked(mockFs.fs.existsSync).mockResolvedValue(true)
    
    await expect(engine.delete(modelId)).resolves.not.toThrow()
  })

  it('should get loaded models', async () => {
    const loadedModels = await engine.getLoadedModels()
    expect(Array.isArray(loadedModels)).toBe(true)
  })

  it('should abort import', async () => {
    // Mock download manager
    global.window = {
      core: {
        extensionManager: {
          getByName: vi.fn().mockReturnValue({
            cancelDownload: vi.fn().mockResolvedValue(true)
          })
        }
      }
    } as any

    await expect(engine.abortImport('test-model')).resolves.not.toThrow()
  })

  it('should handle setting updates', () => {
    expect(() => engine.onSettingUpdate('device', 'cpu')).not.toThrow()
    expect(() => engine.onSettingUpdate('precision', 'float32')).not.toThrow()
  })

  it('should detect model types correctly', () => {
    const engine = new JaxInferenceEngine()
    
    // Access private method via any cast for testing
    const detectModelType = (engine as any).detectModelType.bind(engine)
    
    expect(detectModelType('/path/to/flax_model.msgpack')).toBe('flax')
    expect(detectModelType('/path/to/orbax_model')).toBe('orbax')
    expect(detectModelType('/path/to/regular_model.safetensors')).toBe('jax')
  })
})