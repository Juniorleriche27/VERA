import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VERA — Verifiable Analytics Agent",
  description: "Ask business questions. Get verified analytical evidence.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
