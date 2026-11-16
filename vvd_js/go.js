const safeGoFun = {
  // TODO: a链接安全跳转（只对文章页，关于页评论 -- 评论要单独丢到waline回调中）
  NzcheckLink: async (domName) => {
    // 获取文章页非社会分享的a标签
    const links = document.querySelectorAll(domName);
    if (links) {
      // 锚点正则
      let reg = new RegExp(/^[#].*/);
      for (let i = 0; i < links.length; i++) {
        const ele = links[i];
        let eleHref = ele.getAttribute("href"),
          eleIsDownLoad = ele.getAttribute("data-download"),
          eleRel = ele.getAttribute("rel");

        // 如果你的博客添加了Gitter聊天窗，请去掉下方注释 /*|| link[i].className==="gitter-open-chat-button"*/
        // 排除：锚点、上下翻页、按钮类、分类、标签
        if (
          !reg.test(eleHref) &&
          eleRel !== "prev" &&
          eleRel !== "next" &&
          eleRel !== "category" &&
          eleRel !== "tag" &&
          eleHref !== "javascript:void(0);"
        ) {
          // 判断是否下载地址和白名单，是下载拼接 &type=goDown
          if (!(await safeGoFun.NzcheckLocalSite(eleHref)) && !eleIsDownLoad) {
            // encodeURIComponent() URI编码
            ele.setAttribute(
              "href",
              "/pages/go.html?goUrl=" + encodeURIComponent(eleHref)
            );
          } else if (
            !(await safeGoFun.NzcheckLocalSite(eleHref)) &&
            eleIsDownLoad === "goDown"
          ) {
            ele.setAttribute(
              "href",
              "/pages/go.html?goUrl=" + encodeURIComponent(eleHref) + "&type=goDown"
            );
          }
        }
      }
    }
  },
  // 校验白名单，自己博客，local测试
  NzcheckLocalSite: async (url) => {
    try {
      // 白名单地址则不修改href
      const safeUrls = ["localhost:4000", "zywvvd.com", "www.zywvvd.com"];
      let isOthers = false;
      for (let i = 0; i < safeUrls.length; i++) {
        const ele = safeUrls[i];
        if (url.includes(ele)) {
          isOthers = true;
          break;
        }
      }
      return isOthers;
    } catch (err) {
      return true;
    }
  },
};

Object.keys(safeGoFun).forEach((key) => {
  window[key] = safeGoFun[key];
});

// 页面dom加载完成后，文章页不是分享按钮，不是图片灯箱，class类名不含有 not-check-link
// not-check-link 是小波自己设计的约定类名class，用来排除不调用跳转方法的链接
document.addEventListener("DOMContentLoaded", function () {
  window.NzcheckLink(
    ".post-content a:not(.social-share-icon):not(.fancybox):not(.not-check-link)"
  );
});

