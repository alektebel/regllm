"use client";

import { useEffect, useState } from "react";

export default function NotFound() {
  const [path, setPath] = useState("");

  useEffect(() => {
    setPath(window.location.href);
  }, []);

  return (
    <div style={{ padding: 40, fontFamily: "monospace" }}>
      <h1>404 — Page not found</h1>
      <p><b>Requested URL:</b> {path}</p>
      <p><b>Registered routes:</b> /sas, /pipeline, /compliance, /chat, /test</p>
      <p>
        Try going directly to <a href="/test">/test</a> first.
        If that also 404s, the Next.js server is stale — restart it.
      </p>
    </div>
  );
}
