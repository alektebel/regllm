"use client";

import Link from "next/link";

export default function NotFound() {
  return (
    <div style={{ padding: 40, fontFamily: "monospace" }}>
      <h1>404 — Page not found</h1>
      <p>
        Go to <Link href="/diff">/diff</Link>.
      </p>
    </div>
  );
}
