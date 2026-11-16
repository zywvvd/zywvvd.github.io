//加载完成后执行
// 壁纸兜底：外源壁纸失败或 8s 未完成时回退本地随机壁纸，保证 loading 一定摘掉
(function () {
    var bgLoaded = false;
    var fallback = function () {
        if (!bgLoaded) {
            $('#bg').attr('src', 'https://img.zywvvd.com:61213/HexoFiles/snavigation/img/background' + (Math.floor(Math.random() * 10) + 1) + '.webp');
        }
    };
    $('#bg').on('load', function () { bgLoaded = true; }).on('error', fallback);
    setTimeout(fallback, 8000);
})();

window.addEventListener('load', function () {
    //载入动画
    $('#loading-box').attr('class', 'loaded');
    $('#bg').css("cssText", "transform: scale(1);filter: blur(0px);transition: ease 1.5s;");
    $('#section').css("cssText", "opacity: 1;transition: ease 1.5s;");
    $('.cover').css("cssText", "opacity: 1;transition: ease 1.5s;");

    //用户欢迎
    iziToast.settings({
        timeout: 3000,
        backgroundColor: '#ffffff40',
        titleColor: '#efefef',
        messageColor: '#efefef',
        progressBar: false,
        close: false,
        closeOnEscape: true,
        position: 'topCenter',
        transitionIn: 'bounceInDown',
        transitionOut: 'flipOutX',
        displayMode: 'replace',
        layout: '1'
    });
    setTimeout(function () {
        iziToast.show({
            title: hello,
            message: '欢迎来到 Snavigation'
        });
    }, 800);

    //中文字体缓加载-此处写入字体源文件
    //先行加载简体中文子集，后续补全字集
    //由于压缩过后的中文字体仍旧过大，可转移至对象存储或 CDN 加载
    const font = new FontFace(
        "MiSans",
        "url(" + "https://img.zywvvd.com:61213/HexoFiles/snavigation/font/MiSans-Regular.woff2" + ")"
    );
    document.fonts.add(font);

}, false)

//进入问候
now = new Date(), hour = now.getHours()
if (hour < 6) {
    var hello = "凌晨好";
} else if (hour < 9) {
    var hello = "早上好";
} else if (hour < 12) {
    var hello = "上午好";
} else if (hour < 14) {
    var hello = "中午好";
} else if (hour < 17) {
    var hello = "下午好";
} else if (hour < 19) {
    var hello = "傍晚好";
} else if (hour < 22) {
    var hello = "晚上好";
} else {
    var hello = "夜深了";
}

//获取时间
var t = null;
t = setTimeout(time, 1000);

function time() {
    clearTimeout(t);
    dt = new Date();
    var mm = dt.getMonth() + 1;
    var d = dt.getDate();
    var weekday = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
    var day = dt.getDay();
    var h = dt.getHours();
    var m = dt.getMinutes();
    if (h < 10) {
        h = "0" + h;
    }
    if (m < 10) {
        m = "0" + m;
    }
    $("#time_text").html(h + '<span id="point">:</span>' + m);
    $("#day").html(mm + "&nbsp;月&nbsp;" + d + "&nbsp;日&nbsp;" + weekday[day]);
    t = setTimeout(time, 1000);
}

//获取天气
//原 yiketianqi/tianqiapi 免费接口已停摆(2026-11 实测 15s 无响应)——改 Open-Meteo(无 key 无限额)：
//geojs IP 定位直接返回坐标(CORS *) → open-meteo 天气预报。任一步 6s 超时失败则保持 N/A，不拖页面。
(function () {
    var WMO = { 0:'晴', 1:'多云', 2:'少云', 3:'阴', 45:'雾', 48:'雾凇',
        51:'毛毛雨', 53:'毛毛雨', 55:'毛毛雨', 56:'冻毛毛雨', 57:'冻毛毛雨',
        61:'小雨', 63:'中雨', 65:'大雨', 66:'冻雨', 67:'冻雨',
        71:'小雪', 73:'中雪', 75:'大雪', 77:'米雪',
        80:'阵雨', 81:'阵雨', 82:'强阵雨', 85:'阵雪', 86:'阵雪',
        95:'雷暴', 96:'雷暴夹冰雹', 99:'雷暴夹冰雹' };
    var timed = function (ms) {
        var c = new AbortController();
        setTimeout(function () { c.abort(); }, ms);
        return c.signal;
    };
    fetch('https://get.geojs.io/v1/ip/geo.json', { signal: timed(6000) })
        .then(function (r) { return r.json(); })
        .then(function (geo) {
            if (!geo.latitude || !geo.longitude) throw new Error('no geo');
            return fetch('https://api.open-meteo.com/v1/forecast?latitude=' + geo.latitude + '&longitude=' + geo.longitude +
                '&current=weather_code&daily=temperature_2m_max,temperature_2m_min&forecast_days=1&timezone=auto', { signal: timed(6000) })
                .then(function (r) { return r.json(); });
        })
        .then(function (w) {
            var code = w.current && w.current.weather_code;
            $('#wea_text').text(WMO[code] !== undefined ? WMO[code] : 'N/A');
            $('#tem1').text(Math.round(w.daily.temperature_2m_max[0]));
            $('#tem2').text(Math.round(w.daily.temperature_2m_min[0]));
        })
        .catch(function (e) { console.warn('weather unavailable:', e && e.message || e); });
})();
    
//Tab书签页
$(function () {
    $(".mark .tab .tab-item").click(function () {
        $(this).addClass("active").siblings().removeClass("active");
        $(".products .mainCont").eq($(this).index()).css("display", "flex").siblings().css("display", "none");
    })
})

//设置
$(function () {
    $(".set .tabs .tab-items").click(function () {
        $(this).addClass("actives").siblings().removeClass("actives");
        $(".productss .mainConts").eq($(this).index()).css("display", "flex").siblings().css("display", "none");
    })
})

//输入框为空时阻止跳转
$(window).keydown(function (e) {
    var key = window.event ? e.keyCode : e.which;
    if (key.toString() == "13") {
        if ($(".wd").val() == "") {
            return false;
        }
    }
});

//点击搜索按钮
$(".sou-button").click(function () {
    if ($("body").attr("class") === "onsearch") {
        if ($(".wd").val() != "") {
            $("#search-submit").click();
        }
    }
});

//鼠标中键点击事件
$(window).mousedown(function (event) {
    if (event.button == 1) {
        $("#time_text").click();
    }
});

//控制台输出
var styleTitle1 = `
font-size: 20px;
font-weight: 600;
color: rgb(244,167,89);
`
var styleTitle2 = `
font-size:12px;
color: rgb(244,167,89);
`
var styleContent = `
color: rgb(30,152,255);
`
var title1 = 'Snavigation'
var title2 = `
 _____ __  __  _______     ____     __
|_   _|  \\/  |/ ____\\ \\   / /\\ \\   / /
  | | | \\  / | (___  \\ \\_/ /  \\ \\_/ / 
  | | | |\\/| |\\___ \\  \\   /    \\   /  
 _| |_| |  | |____) |  | |      | |   
|_____|_|  |_|_____/   |_|      |_|                                                     
`
var content = `
版 本 号：1.1
更新日期：2022-07-12

Github:  https://github.com/imsyy/Snavigation
`
console.log(`%c${title1} %c${title2}
%c${content}`, styleTitle1, styleTitle2, styleContent)