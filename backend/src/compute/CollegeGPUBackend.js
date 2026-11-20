import { ComputeBackend } from "./ComputeBackend.js";

/**
 * CollegeGPUBackend — primary live backend (Ch. 71.16, "PRIMARY").
 * Credentials for this backend must never be exposed to the frontend
 * (Ch. 71.17) — they belong in server-side environment config only.
 * Stub: the working implementation is in core/compute (Python).
 */
export class CollegeGPUBackend extends ComputeBackend {
  async getStatus() {
    return { name: "college-gpu", available: false };
  }
}
