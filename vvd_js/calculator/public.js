function OnEnter( field ) { if( field.value == field.defaultValue ) { field.value = ""; } }
function OnExit( field ) { if( field.value == "" ) { field.value = field.defaultValue; } }
function Empty(){document.getElementById('inputstr').value='';}
function copy(ob){var obj=findObj(ob);if(obj){obj.select();try{if(document.execCommand("copy",false,null)){document.execCommand("Copy");document.getElementById("handleResult").innerHTML="复制成功"}else{document.getElementById("handleResult").innerHTML="复制失败，请手动复制"}}catch(err){document.getElementById("handleResult").innerHTML="复制失败，请手动复制"}setTimeout(function(){document.getElementById("handleResult").innerHTML=""},1000)}}function paste(ob){var obj=findObj(ob);if(obj){obj.select();try{if(document.execCommand("paste",false,null)){document.execCommand("Paste");document.getElementById("handleResult").innerHTML="粘贴成功"}else{document.getElementById("handleResult").innerHTML="粘贴失败，请手动粘贴"}}catch(err){document.getElementById("handleResult").innerHTML="粘贴失败，请手动粘贴"}setTimeout(function(){document.getElementById("handleResult").innerHTML=""},1000)}}function cut(ob){var obj=findObj(ob);if(obj){obj.select();try{if(document.execCommand("cut",false,null)){document.execCommand("Cut");document.getElementById("handleResult").innerHTML="剪切成功"}else{document.getElementById("handleResult").innerHTML="剪切失败，请手动剪切"}}catch(err){document.getElementById("handleResult").innerHTML="剪切失败，请手动剪切"}setTimeout(function(){document.getElementById("handleResult").innerHTML=""},1000)}}function findObj(n,d){var p,i,x;if(!d){d=document}if((p=n.indexOf("?"))>0&&parent.frames.length){d=parent.frames[n.substring(p+1)].document;n=n.substring(0,p)}if(!(x=d[n])&&d.all){x=d.all[n]}for(i=0;!x&&i<d.forms.length;i++){x=d.forms[i][n]}for(i=0;!x&&d.layers&&i<d.layers.length;i++){x=findObj(n,d.layers[i].document)}if(!x&&document.getElementById){x=document.getElementById(n)}return x}function qqPYStr(){return""}function traditionalized(cc){var str="";var $language=document.getElementsByName("language");var language="";if($language.length>0){language=$language[1].checked?"zh_TW":""};if(language=="zh_TW"){for(var i in zh_TW){if(cc.indexOf(i)>-1){cc=cc.replace(new RegExp(i,"g"),zh_TW[i])}}}for(var i=0;i<cc.length;i++){if(charPYStr().indexOf(cc.charAt(i))!=-1){str+=ftPYStr().charAt(charPYStr().indexOf(cc.charAt(i)))}else{if(qqPYStr().indexOf(cc.charAt(i))!=-1){str+=ftPYStr().charAt(qqPYStr().indexOf(cc.charAt(i)))}else{str+=cc.charAt(i)}}}return str}function simplized(cc){var str="";for(var i=0;i<cc.length;i++){if(ftPYStr().indexOf(cc.charAt(i))!=-1){str+=charPYStr().charAt(ftPYStr().indexOf(cc.charAt(i)))}else{if(qqPYStr().indexOf(cc.charAt(i))!=-1){str+=charPYStr().charAt(qqPYStr().indexOf(cc.charAt(i)))}else{str+=cc.charAt(i)}}}for(i in toSimplified){if(str.indexOf(i)>-1){str=str.replace(new RegExp(i,"g"),toSimplified[i])}}return str}function convert(nOption){if(nOption==0){document.getElementById('inputstr').value=simplized(document.getElementById('inputstr').value)}else{if(nOption==2){document.getElementById('inputstr').value=qqlized(document.getElementById('inputstr').value)}else{document.getElementById('inputstr').value=traditionalized(document.getElementById('inputstr').value)}}};

function banner_ad() {}

function ad_word_link() {}


function ad_wap_area() {
	if(isMobile()){
		
	}
}

function ad_area_pc_300X250() {
	if(!isMobile()){
		
	}
}

function bookmark(){
    if (document.all){   
        try{   
            window.external.addFavorite(window.location.href,document.title);   
        }catch(e){   
            alert( "您的浏览器不支持自动加入收藏，请使用Ctrl+D进行添加" );   
        }    
    }else if (window.sidebar){   
        window.sidebar.addPanel(document.title, window.location.href, "");   
     }else{   
        alert( "您的浏览器不支持自动加入收藏，请使用Ctrl+D进行添加" );   
    }
}


//判断手机还是电脑,true是手机;
function isMobile(){
	if(/Android|webOS|iPhone|iPod|BlackBerry/i.test(navigator.userAgent)) {
		return true;
	} else {
		return false;
	}
}

//统计
function baidutj() {

}



//首页 180*130 pc
function advertIndexHot01() {
	if(!isMobile()){
		
	}
}

//首页 180*130 pc
function advertIndexHot02() {
	if(!isMobile()){
		
	}
}

function advertIndexMobile() {
	// 240-wap-图文
	if(isMobile()){
		
	}
}


function otherShebaoTools() {
	// 240-wap-图文
	if(isMobile()){
		
		
	} else {
		
	}
}

function advertContentTop() {

}

function advertContentBottom() {

}


function otherToolsZhuanti() {
		
}


function otherTools() {
	
}



function toolsList(){

}


function advertRightSidebar(){

}

document.addEventListener('DOMContentLoaded', function () { var $navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0); if ($navbarBurgers.length > 0) { $navbarBurgers.forEach(function ($el) { $el.addEventListener('click', function () { var target = $el.dataset.target; var $target = document.getElementById(target); $el.classList.toggle('is-active'); $target.classList.toggle('is-active'); }); }); } }); 

//document.oncontextmenu = function(){return false;}
//document.oncontextmenu = function(){event.returnValue = false;}
//document.onselectstart = function(){event.returnValue = false;}
//document.oncopy = function(){event.returnValue = false;}
document.onmousedown = function(e){if( e.which==3 ){return false;}}　// 鼠标右键
document.onkeydown = function(){if( event.keyCode == 123 ){return false;}}

