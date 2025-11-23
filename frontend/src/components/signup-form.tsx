"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Eye, EyeOff, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { getBackendUrl } from "@/lib/env";
import { toast } from "sonner";

const formSchema = z.object({
    email: z.string().email({ message: "Please enter a valid email address" }),
    username: z.string().min(2, { message: "Username must be at least 2 characters" }),
    full_name: z.string().min(2, { message: "Full name must be at least 2 characters" }),
    password: z.string().min(8, { message: "Password must be at least 8 characters" }),
});

type FormValues = z.infer<typeof formSchema>;

export function SignUpPage() {
    const router = useRouter();
    const form = useForm<FormValues>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            email: "",
            username: "",
            full_name: "",
            password: "",
        },
    });

    const [showPassword, setShowPassword] = React.useState(false);
    const [isSubmitting, setIsSubmitting] = React.useState(false);
    const [errorMsg, setErrorMsg] = React.useState<string | null>(null);
    const [redirectCountdown, setRedirectCountdown] = React.useState<number | null>(null);

    React.useEffect(() => {
        if (redirectCountdown == null) return;
        if (redirectCountdown <= 0) {
            router.push("/login");
            return;
        }
        const timer = setTimeout(() => setRedirectCountdown((c) => (c ?? 0) - 1), 1000);
        return () => clearTimeout(timer);
    }, [redirectCountdown, router]);

    async function onSubmit(values: FormValues) {
        setErrorMsg(null);
        setIsSubmitting(true);
        try {
            const backendUrl = getBackendUrl();
            const res = await fetch(`${backendUrl}/api/v1/users/`, {
                method: "POST",
                headers: {
                    accept: "application/json",
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    email: values.email,
                    username: values.username,
                    full_name: values.full_name,
                    password: values.password,
                }),
            });

            if (!res.ok) {
                let message = `Sign up failed (${res.status})`;
                try {
                    const data = await res.json();
                    if (data?.detail) message = Array.isArray(data.detail) ? data.detail[0]?.msg ?? message : data.detail;
                    if (data?.message) message = data.message;
                } catch (_) { }
                setErrorMsg(message);
                return;
            }

            const data = await res.json();
            console.log("Sign up success:", data);
            // Clear form, show toast, and start redirect countdown
            form.reset({ email: "", username: "", full_name: "", password: "" });
            toast("Successfully signed up", { description: "Redirecting to sign in in 3s..." });
            setRedirectCountdown(3);
        } catch (err) {
            console.error(err);
            const message = err instanceof Error ? err.message : "Network error. Please try again.";
            setErrorMsg(message);
        } finally {
            setIsSubmitting(false);
        }
    }

    return (
        <main className="flex min-h-screen items-center justify-center p-4">
            <Card className="w-full max-w-sm">
                <CardHeader>
                    <CardTitle>Sign Up</CardTitle>
                    <CardDescription>Create a new account.</CardDescription>
                </CardHeader>
                <CardContent>
                    <Form {...form}>
                        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                            <FormField
                                control={form.control}
                                name="email"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Email</FormLabel>
                                        <FormControl>
                                            <Input type="email" placeholder="you@example.com" {...field} />
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="username"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Username</FormLabel>
                                        <FormControl>
                                            <Input type="text" placeholder="yourusername" {...field} />
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="full_name"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Full Name</FormLabel>
                                        <FormControl>
                                            <Input type="text" placeholder="Your Name" {...field} />
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
                                                    placeholder={showPassword ? "yourpassword" : "••••••••"}
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

                            <Button type="submit" className="w-full" disabled={isSubmitting || redirectCountdown != null}>
                                {isSubmitting ? (
                                    <span className="inline-flex items-center">
                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        Signing up...
                                    </span>
                                ) : (
                                    "Sign up"
                                )}
                            </Button>

                            {redirectCountdown != null ? (
                                <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    <span>Redirecting in {redirectCountdown}s...</span>
                                </div>
                            ) : null}
                        </form>
                    </Form>
                </CardContent>
                <CardFooter className="flex items-center justify-center">
                    <p className="text-sm text-muted-foreground">
                        Already have an account? {" "}
                        <Link href="/login" className="underline">Login</Link>
                    </p>
                </CardFooter>
            </Card>
        </main>
    );
}