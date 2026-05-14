"use client";

import { useCallback } from "react";
import { authApi } from "@/lib/api";

export function useAuth() {
  const login = useCallback(async (email: string, password: string) => {
    const data = await authApi.login(email, password);
    localStorage.setItem("token", data.access_token);
    // Also store in cookie so Next.js middleware can read it (SSR)
    document.cookie = `token=${data.access_token}; path=/; max-age=${7 * 24 * 3600}; SameSite=Strict`;
  }, []);

  const register = useCallback(async (email: string, password: string) => {
    const data = await authApi.register(email, password);
    localStorage.setItem("token", data.access_token);
    document.cookie = `token=${data.access_token}; path=/; max-age=${7 * 24 * 3600}; SameSite=Strict`;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    document.cookie = "token=; path=/; max-age=0";
    window.location.href = "/login";
  }, []);

  const getToken = useCallback(() => {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("token");
  }, []);

  return { login, register, logout, getToken };
}
