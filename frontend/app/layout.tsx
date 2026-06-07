import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "RegLLM — SAS Diff Explainer",
  description:
    "Why does this field differ between table V2 and V3? Path-integrated " +
    "gradients + Shapley values, grounded by a change-log GraphRAG.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
