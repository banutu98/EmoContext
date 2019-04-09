function getRandText(data) {
  var item = data[Math.floor(Math.random()*data.length)];
  console.log(data);
  console.log(item);
  document.getElementById("replica1").value=item[0];
  document.getElementById("replica2").value=item[1];
  document.getElementById("replica3").value=item[2];
}

function getEmotion() {
  console.log("ok");
}


//document.getElementById("happyBtn").addEventListener("click", happyBtnDisplay);
//document.getElementById("sadBtn").addEventListener("click", sadBtnDisplay);
//document.getElementById("angryBtn").addEventListener("click", angryBtnDisplay);
//document.getElementById("otherBtn").addEventListener("click", otherBtnDisplay);
//
//function happyBtnDisplay() {
//  document.getElementById("replica1").value="How are you?";
//  document.getElementById("replica2").value="I am ok";
//  document.getElementById("replica3").value="me too :)";
//}
//
//function sadBtnDisplay() {
//  document.getElementById("replica1").value="bad :(";
//  document.getElementById("replica2").value="Bad bad! That's the bad kind of bad.";
//  document.getElementById("replica3").value="i know";
//}
//
//function angryBtnDisplay() {
//  document.getElementById("replica1").value="i hate you!";
//  document.getElementById("replica2").value="i will not text you again";
//  document.getElementById("replica3").value="so what?";
//}
//
//function otherBtnDisplay() {
//  document.getElementById("replica1").value="hei";
//  document.getElementById("replica2").value="I am back";
//  document.getElementById("replica3").value="how are you?";
//}
//
//document.getElementById("submit").addEventListener("click", getEmotion);
//function getEmotion() {
//  console.log("ok");
//}