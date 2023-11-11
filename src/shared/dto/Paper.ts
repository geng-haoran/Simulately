import { PaperTagType } from "@site/src/shared/constants/PaperConsts";

export type Paper = {
    title: string;
    description: string;
    preview: string | null; // null = use our serverless screenshot service
    website: string;
    tags: PaperTagType[];
};
