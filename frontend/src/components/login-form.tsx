"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Eye, EyeOff } from "lucide-react";
import { getBackendUrl } from "@/lib/env";

// shadcn ui components (to be added via MCP tools)
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import {
    Form,
    FormControl,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
} from "@/components/ui/form";

const formSchema = z.object({
    username: z.string().min(2, { message: "Username must be at least 2 characters" }),
    password: z.string().min(8, { message: "Password must be at least 8 characters" }),
});

type FormValues = z.infer<typeof formSchema>;

export function LoginPage() {
    const router = useRouter();
    const form = useForm<FormValues>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            username: "",
            password: "",
        },
    });

    const [showPassword, setShowPassword] = React.useState(false);
    const [isSubmitting, setIsSubmitting] = React.useState(false);
    const [errorMsg, setErrorMsg] = React.useState<string | null>(null);

    async function onSubmit(values: FormValues) {
        setErrorMsg(null);
        setIsSubmitting(true);
        try {
            const backendUrl = getBackendUrl();
            const res = await fetch(`${backendUrl}/api/v1/users/login`, {
                method: "POST",
                headers: {
                    accept: "application/json",
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    username: values.username,
                    password: values.password,
                }),
            });

            if (!res.ok) {
                let message = `Login failed (${res.status})`;
                try {
                    const data = await res.json();
                    if (data?.detail) message = Array.isArray(data.detail) ? data.detail[0]?.msg ?? message : data.detail;
                    if (data?.message) message = data.message;
                } catch (_) { }
                setErrorMsg(message);
                return;
            }

            const data = await res.json();
            console.log("Login success:", data);
            // Persist login state (no JWT in API yet, store user profile)
            try {
                const usernameVal = (data?.username ?? data?.user?.username ?? values.username) as string;
                const emailVal = (data?.email ?? data?.user?.email ?? "") as string;
                localStorage.setItem("is_logged_in", "true");
                if (usernameVal) localStorage.setItem("username", usernameVal);
                if (emailVal) localStorage.setItem("user_email", emailVal);
                localStorage.setItem("auth_user", JSON.stringify(data));
            } catch (_) { }

            // Redirect to authenticated home
            router.push("/home");
        } catch (err: any) {
            console.error(err);
            setErrorMsg(err?.message ?? "Network error. Please try again.");
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <main className="flex min-h-screen items-center justify-center p-4">
            <Card className="w-full max-w-sm">
                <CardHeader>
                    <div className="mt-5 mb-5">
                        <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            onClick={() => router.push("/")}
                        >
                            Back to Home
                        </Button>
                    </div>
                    <CardTitle>Login with your Account</CardTitle>
                    {/* <CardDescription>Enter your username and password to sign in.</CardDescription> */}

                </CardHeader>
                <CardContent>
                    <Form {...form}>
                        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                            <FormField
                                control={form.control}
                                name="username"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Username</FormLabel>
                                        <FormControl>
                                            <Input type="text" placeholder="username" {...field} />
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="password"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Password</FormLabel>
                                        <FormControl>
                                            <div className="relative">
                                                <Input
                                                    type={showPassword ? "text" : "password"}
                                                    placeholder={showPassword ? "password" : "••••••••"}
                                                    className="pr-10"
                                                    {...field}
                                                />
                                                <button
                                                    type="button"
                                                    aria-label={showPassword ? "Hide password" : "Show password"}
                                                    onClick={() => setShowPassword((prev) => !prev)}
                                                    className="absolute right-2 top-2 rounded-md p-1 text-muted-foreground hover:text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                                                >
                                                    {showPassword ? (
                                                        <EyeOff className="h-4 w-4" />
                                                    ) : (
                                                        <Eye className="h-4 w-4" />
                                                    )}
                                                </button>
                                            </div>
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            {errorMsg ? (
                                <p className="text-sm text-destructive" role="alert">{errorMsg}</p>
                            ) : null}
                            <Button type="submit" className="w-full" disabled={isSubmitting}>
                                {isSubmitting ? "Signing in..." : "Sign in"}
                            </Button>
                        </form>
                    </Form>
                </CardContent>
                <CardFooter className="flex items-center justify-center">
                    <p className="text-sm text-muted-foreground">
                        Don&apos;t have an account? {" "}
                        <Link href="/signup" className="underline">Sign up</Link>
                    </p>
                </CardFooter>
            </Card>
        </main>
    );
}