(this.webpackJsonpfrontend=this.webpackJsonpfrontend||[]).push([[0],{100:function(e,t,c){"use strict";c.r(t);var s=c(2),a=c(0),n=c.n(a),r=c(20),l=c.n(r),o=(c(71),c(17)),j=c(14),i=c(7),d=c(108),h=c(110),b=c(107),u=c(109),O=c(57),x=c(56);var p=Object(i.i)((function(e){return Object(s.jsxs)(d.a,{bg:"light",expand:"lg",children:[Object(s.jsx)(d.a.Brand,{href:"/",children:"\u0410\u0440\u0445\u0438\u0432\u0430\u0440\u0438\u0443\u0441"}),Object(s.jsx)(d.a.Toggle,{"aria-controls":"basic-navbar-nav"}),Object(s.jsxs)(d.a.Collapse,{id:"basic-navbar-nav",children:[Object(s.jsxs)(h.a,{className:"mr-auto",children:[Object(s.jsx)(h.a.Link,{href:"/search",children:"\u041f\u043e\u0438\u0441\u043a"}),Object(s.jsx)(h.a.Link,{href:"/upload",children:"\u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430"}),Object(s.jsxs)(b.a,{title:"\u041e\u0442\u0447\u0451\u0442\u044b",id:"basic-nav-dropdown",children:[Object(s.jsx)(b.a.Item,{href:"/stats/system",children:"\u0421\u0438\u0441\u0442\u0435\u0441\u0442\u0435\u043c\u0430"}),Object(s.jsx)(b.a.Item,{href:"/stats/bi",children:"BI"})]})]}),Object(s.jsxs)(u.a,{inline:!0,method:"GET",action:"/search",children:[Object(s.jsx)(O.a,{type:"text",name:"text",placeholder:"\u041f\u043e\u0438\u0441\u043a \u043f\u043e \u0442\u0435\u043a\u0441\u0442\u0443",className:"mr-sm-2"}),Object(s.jsx)(x.a,{type:"submit",variant:"outline-success",children:"\u0418\u0441\u043a\u0430\u0442\u044c"})]})]})]})})),m=function(){return Object(s.jsx)("div",{children:" "})},f=c(29),v=c(60),g=c(61),y=c(32),C=c(65),_=c(64),w=c(42),I=c.n(w),k=c(62),N=c(63),S=c.n(N),F="http://178.154.225.145:5000";function G(e,t){return L.apply(this,arguments)}function L(){return(L=Object(k.a)(I.a.mark((function e(t,c){var s,a,n=arguments;return I.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return s=n.length>2&&void 0!==n[2]?n[2]:"application/json",e.prev=1,e.next=4,S.a.post(F+"/api/"+t,c,{headers:{"Content-Type":s}});case 4:a=e.sent,console.log(a),a.data.status=a.data&&a.data.status||"ok",e.next=13;break;case 9:return e.prev=9,e.t0=e.catch(1),console.log(e.t0),e.abrupt("return",{status:"error"});case 13:return console.log(a.data),e.abrupt("return",a.data);case 15:case"end":return e.stop()}}),e,null,[[1,9]])})))).apply(this,arguments)}var T=function(e){Object(C.a)(c,e);var t=Object(_.a)(c);function c(e){var s;return Object(v.a)(this,c),(s=t.call(this,e)).state={},s.handleChange=s.handleChange.bind(Object(y.a)(s)),s.onUpload=s.onUpload.bind(Object(y.a)(s)),s}return Object(g.a)(c,[{key:"handleChange",value:function(e){var t=e.target,c=t.name;console.log(t),"file"!==c?this.setState(Object(f.a)({},c,t.value)):this.setState(Object(f.a)({},c,t.files[0]))}},{key:"onUpload",value:function(e){var t=this;e.preventDefault();var c=new FormData;c.append("file",this.state.file),G("upload",c,"multipart/form-data").then((function(e){console.log(e),"ok"===e.status&&t.props.history.push("/upload/"+e.upload_id)}))}},{key:"render",value:function(){return Object(s.jsxs)("div",{children:[Object(s.jsx)("h1",{className:"mt-3",children:"\u0417\u0430\u0433\u0440\u0443\u0437\u043a\u0430 \u0444\u0430\u0439\u043b\u0430"}),Object(s.jsxs)(u.a,{type:"POST",onSubmit:this.onUpload,children:[Object(s.jsx)(u.a.File,{id:"custom-file",label:"\u0412\u044b\u0431\u0435\u0440\u0435\u0442\u0435 pdf \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u0438\u043b\u0438 zip-\u0430\u0440\u0445\u0438\u0432 \u0441 pdf \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u043c\u0438",name:"file",onChange:this.handleChange}),Object(s.jsx)(x.a,{className:"mt-3",type:"submit",variant:"outline-success",children:"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c"})]})]})}}]),c}(n.a.Component),E=Object(i.i)(T),U=c(105),z=c(58);var B=function(e){return console.log(e),Object(s.jsxs)(u.a.Group,{as:U.a,controlId:"exampleForm.ControlSelect1",children:[Object(s.jsx)(u.a.Label,{column:!0,sm:"2",children:"\u041a\u043b\u0430\u0441\u0441 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430"}),Object(s.jsx)(z.a,{sm:"10",children:Object(s.jsxs)(u.a.Control,{onChange:e.handleChange,as:"select",name:"doc_cls",value:e.type,children:[Object(s.jsx)("option",{value:"undefined",children:"\u041b\u044e\u0431\u043e\u0439"}),e.classes.map((function(e,t){return Object(s.jsx)("option",{value:e,children:e})}))]})})]})},D=function(e){var t=Object(i.h)().docId,c=Object(a.useState)({}),n=Object(o.a)(c,2),r=n[0],l=n[1];return"failed"!==r.process_status&&"complete"!==r.process_status&&setTimeout((function(){G("view/"+t,{},"").then((function(e){l(e)}))}),2e3),"complete"===r.process_status?Object(s.jsxs)("div",{children:[Object(s.jsxs)("h1",{className:"mt-2",children:[" \u041f\u0440\u043e\u0441\u043c\u043e\u0442\u0440 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430 ",r.doc_src_name," "]}),Object(s.jsx)("hr",{}),Object(s.jsxs)(u.a,{method:"GET",children:[Object(s.jsx)(B,{classes:e.classes,handleChange:null,type:r.doc_cls}),Object(s.jsxs)(u.a.Group,{as:U.a,controlId:"exampleForm.ControlInput1",children:[Object(s.jsx)(u.a.Label,{column:!0,sm:"2",children:"\u041d\u043e\u043c\u0435\u0440 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430"}),Object(s.jsx)(z.a,{sm:"10",children:Object(s.jsx)(u.a.Control,{type:"text",name:"number",placeholder:"\u041d\u043e\u043c\u0435\u0440 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430",value:r.number})})]}),Object(s.jsxs)(u.a.Group,{as:U.a,sm:"5",controlId:"exampleForm.DateFrom",children:[Object(s.jsx)(u.a.Label,{column:!0,sm:"2",children:"\u0414\u0430\u0442\u0430 \u043f\u043e\u0434\u043f\u0438\u0441\u0430\u043d\u0438\u044f"}),Object(s.jsx)(z.a,{sm:"10",children:Object(s.jsx)(u.a.Control,{type:"date"})})]}),Object(s.jsxs)(u.a.Group,{as:U.a,controlId:"exampleForm.ControlInput2",children:[Object(s.jsx)(u.a.Label,{column:!0,sm:"2",children:"\u0412\u044b\u0434\u0430\u0432\u0448\u0438\u0439 \u043e\u0440\u0433\u0430\u043d"}),Object(s.jsx)(z.a,{sm:"10",children:Object(s.jsx)(u.a.Control,{type:"text",name:"organization",placeholder:"\u0412\u044b\u0434\u0430\u0432\u0448\u0438\u0439 \u043e\u0440\u0433\u0430\u043d",value:r.organization})})]}),Object(s.jsx)(x.a,{type:"submit",variant:"outline-success",children:"\u0421\u043e\u0445\u0440\u0430\u043d\u0438\u0442\u044c \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u044f"})]}),Object(s.jsx)("hr",{}),Object(s.jsx)("iframe",{className:"col-12",style:{height:"800px"},src:"http://178.154.225.145:5000/api/pdf/"+t})]}):"failed"===r.process_status?Object(s.jsx)("h1",{children:"\u041f\u0440\u043e\u0438\u0437\u043e\u0448\u043b\u0430 \u043e\u0448\u0438\u0431\u043a\u0430 \u043f\u0440\u0438 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0435 \u0434\u0430\u043d\u043d\u044b\u0445"}):Object(s.jsx)("h1",{children:"\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430..."})},P=c(106);var J=function(e){return Object(s.jsxs)("div",{children:[Object(s.jsxs)(U.a,{className:"mt-5",children:[Object(s.jsxs)(z.a,{className:"col-10",children:[Object(s.jsx)("h2",{children:"\u0420\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b:"})," "]}),Object(s.jsx)(z.a,{className:"col-2",children:Object(s.jsx)(x.a,{variant:"outline-success",children:"\u042d\u043a\u0441\u043f\u043e\u0440\u0442 \u0432 Excell"})})]}),Object(s.jsxs)(P.a,{striped:!0,bordered:!0,hover:!0,size:"sm",children:[Object(s.jsx)("thead",{children:Object(s.jsxs)("tr",{children:[Object(s.jsx)("th",{children:"id"}),Object(s.jsx)("th",{children:"\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435"}),Object(s.jsx)("th",{children:"\u041a\u043b\u0430\u0441\u0441"})]})}),Object(s.jsx)("tbody",{children:e.result.map((function(e,t){return Object(s.jsxs)("tr",{children:[Object(s.jsx)("td",{children:Object(s.jsx)(j.b,{to:"/view/"+e.doc_id,children:e.doc_id})}),Object(s.jsx)("td",{children:Object(s.jsx)(j.b,{to:"/view/"+e.doc_id,children:e.doc_src_name})}),Object(s.jsx)("td",{children:Object(s.jsx)(j.b,{to:"/view/"+e.doc_id,children:e.doc_cls})})]})}))})]})]})};var W=function(e){var t=Object(a.useState)({}),c=Object(o.a)(t,2),n=c[0],r=c[1],l=Object(a.useState)(null),j=Object(o.a)(l,2),d=j[0],h=j[1],b=Object(a.useState)(null),O=Object(o.a)(b,2),p=(O[0],O[1],function(e){var t=e.target,c=t.name;console.log(t),r("file"!==c?Object(f.a)({},c,t.value):Object(f.a)({},c,t.files[0]))}),m=new URLSearchParams(Object(i.g)().search);return Object(s.jsxs)("div",{children:[Object(s.jsx)("h1",{className:"mt-2",children:" \u041f\u043e\u0438\u0441\u043a "}),Object(s.jsx)("hr",{}),Object(s.jsxs)(u.a,{method:"GET",onSubmit:function(e){e.preventDefault(),G("search",n).then((function(e){console.log(e),"ok"===e.status&&h(e)}))},children:[Object(s.jsxs)(u.a.Group,{as:U.a,controlId:"exampleForm.ControlInput1",children:[Object(s.jsx)(u.a.Label,{column:!0,sm:"2",children:"\u0422\u0435\u043a\u0441\u0442"}),Object(s.jsx)(z.a,{sm:"10",children:Object(s.jsx)(u.a.Control,{onChange:p,type:"text",name:"text",placeholder:"\u0414\u043e\u043a\u0443\u043c\u0435\u043d\u0442 \u0441\u043e\u0434\u0435\u0440\u0436\u0438\u0442...",value:m.get("text")})})]}),Object(s.jsx)(B,{classes:e.classes,handleChange:p,type:m.get("doc_cls")}),Object(s.jsx)(x.a,{type:"submit",variant:"outline-success",children:"\u0418\u0441\u043a\u0430\u0442\u044c"})]}),Object(s.jsx)("hr",{}),null!=d?Object(s.jsx)(J,{result:d}):null]})},A=Object(i.i)((function(e){var t=Object(i.h)().uploadId,c=Object(a.useState)({}),n=Object(o.a)(c,2),r=n[0],l=n[1];return"failed"!==r.process_status&&"complete"!==r.process_status&&setTimeout((function(){G("upload/"+t,{},"").then((function(e){l(e)}))}),2e3),"complete"===r.process_status?Object(s.jsx)("div",{children:Object(s.jsx)(J,{result:r.result})}):"failed"===r.process_status?Object(s.jsx)("h1",{children:"\u041f\u0440\u043e\u0438\u0437\u043e\u0448\u043b\u0430 \u043e\u0448\u0438\u0431\u043a\u0430 \u043f\u0440\u0438 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0435 \u0434\u0430\u043d\u043d\u044b\u0445"}):Object(s.jsx)("h1",{children:"\u041e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0430..."})})),M=function(){var e=Object(a.useState)([]),t=Object(o.a)(e,2),c=t[0],n=t[1];return 0===c.length&&G("info/classes",[]).then((function(e){if("ok"===e.status){var t=[];for(var c in delete e.status,e)e.hasOwnProperty(c)&&t.push(e[c]);n(t)}})),Object(s.jsx)("div",{className:"Main container-md",children:Object(s.jsxs)(j.a,{children:[Object(s.jsx)(p,{}),Object(s.jsxs)(i.d,{children:[Object(s.jsx)(i.b,{exact:!0,path:"/",children:Object(s.jsx)(i.a,{to:"/search"})}),Object(s.jsx)(i.b,{exact:!0,path:"/upload/",children:Object(s.jsx)(E,{})}),Object(s.jsx)(i.b,{exact:!0,path:"/upload/:uploadId",children:Object(s.jsx)(A,{})}),Object(s.jsx)(i.b,{exact:!0,path:"/view/:docId",children:Object(s.jsx)(D,{classes:c})}),Object(s.jsx)(i.b,{path:"/search",children:Object(s.jsx)(W,{classes:c})}),Object(s.jsx)(i.b,{exact:!0,path:"/report"}),Object(s.jsx)(i.b,{path:"*",children:Object(s.jsx)("h1",{children:"Not Found"})})]}),Object(s.jsx)(m,{})]})})},R=(c(96),c(97),function(){return Object(s.jsx)("div",{className:"App",children:Object(s.jsx)(M,{})})});Boolean("localhost"===window.location.hostname||"[::1]"===window.location.hostname||window.location.hostname.match(/^127(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}$/));l.a.render(Object(s.jsx)(R,{}),document.getElementById("root")),"serviceWorker"in navigator&&navigator.serviceWorker.ready.then((function(e){e.unregister()})).catch((function(e){console.error(e.message)}))},71:function(e,t,c){}},[[100,1,2]]]);
//# sourceMappingURL=main.f3242769.chunk.js.map