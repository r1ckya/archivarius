import React from 'react';
import {Button, Col, Form, FormControl, Nav, Navbar, NavDropdown, Row, Table} from "react-bootstrap";
import APIRequest from "../../rest";
import { withRouter } from "react-router-dom";

class Upload extends React.Component {

  constructor(props) {
    super(props);
    this.state = {};

    this.handleChange = this.handleChange.bind(this);
    this.onUpload = this.onUpload.bind(this);
  }
  handleChange(event) {
    const target = event.target;
    const name = target.name;
    console.log(target);
    if (name !== 'file') {
      this.setState({
        [name]: target.value
      });
    } else {
      this.setState({
        [name]: target.files[0]
      });
    }
  }

  onUpload(e) {
    let ths = this;
    e.preventDefault();
    var formData = new FormData();
    formData.append("file", this.state.file);
    APIRequest('upload', formData, "multipart/form-data").then(function (result) {
      console.log(result);
      if (result.status === 'ok') {
        ths.props.history.push('/upload/'+ result.upload_id);
      }
    });

  }

  render() {
    return (
      <div>
        <h1 className={"mt-3"}>Загрузка файла</h1>
        <Form type={"POST"} onSubmit={this.onUpload}>
          <Form.File
            id="custom-file"
            label="Выберете pdf документ или zip-архив с pdf документами"
            name={"file"}
            onChange={this.handleChange}
          />
          <Button className={"mt-3"} type={"submit"} variant="outline-success">Загрузить</Button>
        </Form>
      </div>
    )
  }

}

export default withRouter(Upload);