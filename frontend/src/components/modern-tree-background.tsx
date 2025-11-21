import React, { useEffect, useRef } from 'react';

const ModernTreeBackground: React.FC = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let width = window.innerWidth;
        let height = window.innerHeight;

        const resize = () => {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width;
            canvas.height = height;
        };

        window.addEventListener('resize', resize);
        resize();

        interface Branch {
            x: number;
            y: number;
            length: number;
            angle: number;
            depth: number;
            maxDepth: number;
            width: number;
            color: string;
            growth: number;
            speed: number;
            children: Branch[];
            isFlower: boolean;
        }

        const branches: Branch[] = [];

        const createBranch = (x: number, y: number, length: number, angle: number, depth: number, maxDepth: number): Branch => {
            // Sakura logic:
            // Top levels are flowers (Light Pink), Bottom are trunk (Dark Rose/Pink)
            const isFlower = depth > maxDepth - 3;

            // Width: Taper off, but flowers stay delicate
            const branchWidth = isFlower ? Math.max(0.5, (maxDepth - depth)) : Math.max(1, (maxDepth - depth) * 0.9);

            let color;
            if (isFlower) {
                // Random Pink Variations (Light)
                // R: 255, G: 190-230, B: 200-240
                const g = 180 + Math.floor(Math.random() * 50);
                const b = 190 + Math.floor(Math.random() * 50);
                color = `rgba(255, ${g}, ${b}, ${0.7 + Math.random() * 0.3})`;
            } else {
                // Darker Pink/Rosewood for branches (instead of grey/brown)
                // "Just a bit darker than the flower" implies maintaining the pink hue but lowering lightness.

                // Base Dark Rose: R:160, G:90, B:110
                // We lighten it slightly as it goes up the tree to blend with flowers.
                const r = 160 + (depth * 8);
                const g = 80 + (depth * 8);
                const b = 100 + (depth * 8);

                // Opacity needs to be solid for trunk
                color = `rgba(${r}, ${g}, ${b}, ${0.8 + (depth / maxDepth) * 0.2})`;
            }

            return {
                x,
                y,
                length,
                angle,
                depth,
                maxDepth,
                width: branchWidth,
                color,
                growth: 0,
                speed: 0.04 + Math.random() * 0.04,
                children: [],
                isFlower
            };
        };

        const init = () => {
            branches.length = 0;
            // Density: Middle ground (width / 450 gives ~4-5 trees on desktop)
            const rootCount = Math.max(5, Math.floor(width / 450));

            for (let i = 0; i < rootCount; i++) {
                branches.push(createBranch(
                    (width / (rootCount + 1)) * (i + 1) + (Math.random() * 100 - 75),
                    height,
                    height * 0.16 + Math.random() * 60,
                    -Math.PI / 2 + (Math.random() * 0.3 - 0.15),
                    0,
                    9 // Depth 9: Balanced between sparse (7) and dense (11)
                ));
            }
        };

        init();

        const drawBranch = (branch: Branch) => {
            if (branch.growth < 1) {
                branch.growth += branch.speed;
                if (branch.growth > 1) branch.growth = 1;
            }

            const currentLength = branch.length * branch.growth;
            const endX = branch.x + Math.cos(branch.angle) * currentLength;
            const endY = branch.y + Math.sin(branch.angle) * currentLength;

            ctx.beginPath();
            ctx.moveTo(branch.x, branch.y);
            ctx.lineTo(endX, endY);
            ctx.strokeStyle = branch.color;
            ctx.lineWidth = branch.width;
            ctx.lineCap = 'round';
            ctx.stroke();

            // Bloom effect for flower branches
            if (branch.isFlower && branch.growth > 0.8) {
                ctx.beginPath();
                // Draw a small circle at the end to simulate a petal/flower cluster
                ctx.arc(endX, endY, branch.width * 2.5, 0, Math.PI * 2);
                ctx.fillStyle = branch.color;
                ctx.fill();
            }

            // Spawn children
            if (branch.growth >= 1 && branch.children.length === 0 && branch.depth < branch.maxDepth) {
                const randomVal = Math.random();
                let numChildren = 2;

                // Branching logic adjusted for "Middle" density
                if (randomVal > 0.75) numChildren = 3; // 25% chance of 3
                else if (randomVal < 0.15) numChildren = 1; // 15% chance of 1

                // Flowers tend to cluster more
                if (branch.isFlower && Math.random() > 0.6) {
                    numChildren = 3;
                }

                for (let i = 0; i < numChildren; i++) {
                    // Sakura spread is often wide
                    const angleOffset = (Math.random() - 0.5) * Math.PI / 1.4;
                    const newAngle = branch.angle + angleOffset;
                    const newLength = branch.length * (0.7 + Math.random() * 0.15);
                    branch.children.push(createBranch(endX, endY, newLength, newAngle, branch.depth + 1, branch.maxDepth));
                }
            }

            for (const child of branch.children) {
                drawBranch(child);
            }
        };

        let animationId: number;
        const animate = () => {
            ctx.clearRect(0, 0, width, height);
            for (const branch of branches) {
                drawBranch(branch);
            }

            animationId = requestAnimationFrame(animate);
        };

        animate();

        return () => {
            window.removeEventListener('resize', resize);
            cancelAnimationFrame(animationId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none z-0 opacity-70"
        />
    );
};

export default ModernTreeBackground;