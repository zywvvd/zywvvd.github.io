	

console.log('album js Hello World')

photo ={
    page: 1,
    offset: 20,
    init: function () {
        var that = this;
        $.getJSON("readme.json_", function (data) {
            that.render(that.page, data);
            //that.scroll(data);
        });
    },
    render: function (page, data) {
    	var year = data["time"]["year"]
    	var month = data["time"]["month"]
    	var day = data["time"]["day"]
    	var type = data["type"]
    	var model = data["model"]
    	var city = data["position"]['city']
    	var street = data["position"]['street']
    	var title = data["title"]
    	var balabala = data["balabala"]

    	var image_info_list = data["image_info"]
    	console.log(image_info_list[0]["Image_Model"])

    	console.log(year)
        //var begin = (page - 1) * this.offset;
        //var end = page * this.offset;
        //if (begin >= data.length) return;
        var html, imgNameWithPattern, imgName, imageSize, imageX, imageY, li = "";

		$(".album_image_grid").append('<br>');
		$(".album_image_grid").append('<br>');

        for (var i = 0; i < image_info_list.length ; i++) {
        	sub_image = image_info_list[i]
        	nbsp = "&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp"
        	href_str = ""

        	image_model = sub_image["Image_Model"] //相机型号
        	if (image_model!="")
        	{
        		href_str = href_str+'相机：'+image_model+nbsp
        	}

        	EXIF_fnumber = sub_image["EXIF_FNumber"] //光圈	
        	if (EXIF_fnumber!="")
        	{
        		href_str = href_str+' 光圈：' +EXIF_fnumber +nbsp
        	}

        	EXIF_FocalLength = sub_image["EXIF_FocalLength"] //焦距
        	if (EXIF_FocalLength!="")
        	{
        		href_str = href_str+' 焦距：' +EXIF_FocalLength +nbsp
        	}

        	EXIF_exposureMode = sub_image["EXIF_ExposureMode"] //曝光模式
        	if (EXIF_exposureMode!="")
        	{
        		href_str = href_str+' 曝光模式：' +EXIF_exposureMode +nbsp
        	}		

        	EXIF_exposureTime = sub_image["EXIF_ExposureTime"] //曝光时间
        	if (EXIF_exposureTime!="")
        	{
        		href_str = href_str+' 曝光时间：' +EXIF_exposureTime +nbsp
        	}	

        	EXIF_ISOSpeedRatings = sub_image["EXIF_ISOSpeedRatings"] //ISO
        	if (EXIF_ISOSpeedRatings!="")
        	{
        		href_str = href_str+' ISO：' +EXIF_ISOSpeedRatings +nbsp
        	}

        	image_url = sub_image["url"]

            li += '<div class="card" style="width:100%">' +
                    '<div class="ImageInCard">' +
                      '<a data-fancybox="gallery" href="' + image_url + '?raw=true" data-caption="' + href_str +'">' +
                        '<img src="' + image_url + '?raw=true"/>' +
                      '</a>' + '<br>' +
                    '</div>' +
                    // '<div class="TextInCard">' + imgName + '</div>' +
                  '</div>'
        }
        $(".album_image_grid").append(li);
        //$(".album_image_grid").lazyload();
        //this.minigrid();
    },
    
}
photo.init();
