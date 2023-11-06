export function yyyyMMdd(): string {
    return new Date().toISOString().slice(0, 10).replace(/-/g, '');
}

export function yyyyMMddhhmm(): string {
    return new Date().toISOString().slice(0, 16).replace(/-/g, '').replace(/:/g, '').replace(/T/g, '');
}

/**
 * Returns the screenshot url for a given website
 * Ref: https://api-explorer.11ty.dev/
 * @param url website url
 * @returns image url of the screenshot
 */
export function GetWebsiteScreenshot(url: string): string {

    var slug = yyyyMMdd();

    return `https://v1.screenshot.11ty.dev/${encodeURIComponent(url)}/opengraph/smaller/_${slug}`;

    // return `https://slorber-api-screenshot.netlify.app/${encodeURIComponent(url)}/showcase/_${slug}`;
}