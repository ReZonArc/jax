import { vi } from 'vitest'

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
}

Object.defineProperty(globalThis, 'localStorage', {
  value: localStorageMock,
  writable: true,
})

// Mock the global window object
Object.defineProperty(globalThis, 'window', {
  value: {
    localStorage: localStorageMock,
    core: {
      extensionManager: {
        getByName: vi.fn().mockReturnValue({
          downloadFiles: vi.fn().mockResolvedValue(undefined),
          cancelDownload: vi.fn().mockResolvedValue(undefined),
        }),
      },
    },
  },
})

// Mock global constants that are injected during build
Object.defineProperty(globalThis, 'SETTINGS', {
  value: [],
  writable: true,
})

Object.defineProperty(globalThis, 'ENGINE', {
  value: 'jax-ml',
  writable: true,
})