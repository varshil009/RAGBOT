window.PDFViewerApplicationOptions = {
    // Enable scripting support for the viewer
    enableScripting: true,
    
    // Allow JavaScript evaluation (needed for some viewer features)
    isEvalSupported: true,
    
    // Set external link behavior (2 means open in new tab)
    externalLinkTarget: 2,
    
    // Additional useful configurations
    scrollModeOnLoad: 0,  // 0 = vertical scrolling
    spreadModeOnLoad: 0,  // 0 = single page view
    
    // Enable the toolbar
    toolbar: {
        visible: true,
    },
    
    // Enable sidebar (for thumbnails, outline, etc.)
    sidebar: {
        visible: true,
    },
    
    // Document loading options
    showPreviousViewOnLoad: false,
    defaultZoomValue: 'page-fit',
};