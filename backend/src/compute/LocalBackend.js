import { ComputeBackend } from "./ComputeBackend.js";

/**
 * LocalBackend — Mac MPS/CPU fallback (Ch. 71.16, "LOCAL FALLBACK").
 * Stub: the working implementation is in core/compute (Python).
 */
export class LocalBackend extends ComputeBackend {
  async getStatus() {
    return { name: "local", available: false };
  }
}
