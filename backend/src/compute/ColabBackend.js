import { ComputeBackend } from "./ComputeBackend.js";

/**
 * ColabBackend — Google Colab GPU, used as a temporary/backup backend
 * (Ch. 71.16, "BACKUP"). Stub: the working implementation is in core/compute (Python).
 */
export class ColabBackend extends ComputeBackend {
  async getStatus() {
    return { name: "colab", available: false };
  }
}
