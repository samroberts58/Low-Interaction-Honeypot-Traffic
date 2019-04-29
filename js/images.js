//image rotation

var myImage = document.getElementById("main_image");

var image_array = ['images/Me.jpg', 'images/honey-pot.jpg'];

var image_index = 0;
function changeImage() {
	myImage.setAttribute("src", image_array[image_index]);
	image_index++;
	if (image_index >= image_array.length) {
		image_index = 0;
	}
}

//setInterval is used to create animation
var intervalHandle = setInterval(changeImage, 10000);

myImage.onclick = function() {
	clearInterval(intervalHandle);
}
