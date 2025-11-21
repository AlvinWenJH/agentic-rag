export interface ConceptNode {
    name: string;
    description?: string;
    children?: ConceptNode[];
}

export interface TreeConfig {
    depth: number;
    branchingFactor: number;
}

export enum AppState {
    LANDING = 'LANDING',
    EXPLORER = 'EXPLORER'
}