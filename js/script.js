// Toggle Chart div's
$(document).ready(function(){
  $(".edabar").click(function(){
    $("#edabar").toggle();
  });
});
$(document).ready(function(){
  $(".edascatter").click(function(){
    $("#edascatter").toggle();
  });
});
$(document).ready(function(){
  $(".edainteract").click(function(){
    $("#edainteract").toggle();
  });
});
$(document).ready(function(){
  $(".chi").click(function(){
    $("#chi").toggle();
  });
});
$(document).ready(function(){
  $(".cov").click(function(){
    $("#cov").toggle();
  });
});
$(document).ready(function(){
  $(".poisson").click(function(){
    $("#poisson").toggle();
  });
});
$(document).ready(function(){
  $(".rfc").click(function(){
    $("#rfc").toggle();
  });
});
$(document).ready(function(){
  $(".hyp").click(function(){
    $("#hyp").toggle();
  });
});
$(document).ready(function(){
  $(".best").click(function(){
    $("#best").toggle();
  });
});


// Toggle Sections
var coll = document.getElementsByClassName("toggle");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
    } else {
      content.style.display = "block";
    }
  });
}
