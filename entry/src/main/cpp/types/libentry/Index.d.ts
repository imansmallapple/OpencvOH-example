import resourceManager from '@ohos.resourceManager';

export interface PixelInfo {
  rows: number;
  cols: number;
  buffSize: number;
  byteBuffer: ArrayBuffer;
}

export const add: (a: number, b: number) => number;
export const img2Gray: (resmgr: resourceManager.ResourceManager, path: string, file: string) => PixelInfo;