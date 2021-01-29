console.log('Photos js Hello World')

photo ={
    init: function () {
        var that = this;
        $.getJSON("album.json_", function (data) {
            that.render(that.page, data);
            //that.scroll(data);
        });
    },
    render: function (page, data) {
        var album_list = data['album'];
        var html, imgNameWithPattern, imgName, imageSize, imageX, imageY, li = "";
        var link_profix = "/photos/"
        for (var i = 0; i < album_list.length; i++) {

            album_info = album_list[i]

            dir_name = album_info["directory"]

            title = album_info["title"]
            /*console.log(album_info)
            imgNameWithPattern = data[i].split(' ')[1];
            imgName = imgNameWithPattern.split('.')[0]
            imageSize = data[i].split(' ')[0];
            imageX = imageSize.split('.')[0];
            imageY = imageSize.split('.')[1];*/
            li += '<div>' +
                      '<a href="' +link_profix + dir_name + '/">' +
                          title + '<br>' + 
                      '</a>' +
                    // '<div class="TextInCard">' + imgName + '</div>' +
                  '</div>'
        }
        $(".album_link_list").append(li);
        //$(".ImageGrid").lazyload();
        this.minigrid();
    },
    minigrid: function() {
        var grid = new Minigrid({
            container: '.ImageGrid',
            item: '.card',
            gutter: 12
        });
        grid.mount();
        $(window).resize(function() {
           grid.mount();
        });
    }
}
photo.init();
